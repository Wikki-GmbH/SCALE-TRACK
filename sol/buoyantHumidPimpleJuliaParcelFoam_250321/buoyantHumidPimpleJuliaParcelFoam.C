/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2021 OpenCFD Ltd.
    Copyright (C) 2024 Sergey Lesnik
    Copyright (C) 2024 Henrik Rusche
    Copyright (C) 2024 Silvio Schmalfuß
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    buoyantHumidPimpleJuliaParcelFoam

Group
    grpHeatTransferSolvers

Description
    Test solver augmented from buoyantPimpleFoam for dilute multiphase flows
    using coupling between OpenFOAM and Julia code using Euler-Lagrange ansatz.
    OpenFOAM is responsible for the carrier and Julia for the dispersed phase.
    The Julia file is located in the case directory.

Author
    Sergey Lesnik, Wikki GmbH, 2024
    Henrik Rusche, Wikki GmbH, 2024
    Silvio Schmalfuß, TROPOS, 2024

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "rhoThermo.H"
#include "turbulentFluidThermoModel.H"
#include "radiationModel.H"
#include "CorrectPhi.H"
#include "fvOptions.H"
#include "pimpleControl.H"
#include "pressureControl.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "UPstream.H"
#include <typeinfo>

extern "C"
{
#include "juliaHelper.h"
}

#include "juliaWrapper.H"

#include <julia.h>

// only define this once, in an executable (not in a shared library) if you
// want fast code.
JULIA_DEFINE_FAST_TLS

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    // Initialise MPI with MPI_THREAD_MULTIPLE as required by MPI.jl
    // Ignore the warning later on
    // Initialise only if arguments contain "-parallel"
    for (int argi = 1; argi < argc; ++argi)
    {
        const char *optName = argv[argi];

        if (optName[0] == '-')
        {
            ++optName;  // Looks like an option, skip leading '-'

            if (strcmp(optName, "parallel") == 0)
            {
                UPstream::init(argc, argv, true);
            }
        }
    }

    Info<< "Initialising Julia" << endl;
    int jl_argc = 2;
    char** jl_argv = static_cast<char**>(malloc(sizeof(char*)*jl_argc));
    jl_argv[0] = argv[0];
    string threadsOpt = "--threads=2";
    jl_argv[1] = const_cast<char*>(threadsOpt.c_str());
    jl_parse_opts(&jl_argc, &jl_argv);
    jl_init();
    Info<< "Initialising Julia - done" << endl;
    
    argList::addNote
    (
        "Transient solver for buoyant, turbulent fluid flow"
        " of compressible fluids, including radiation,"
        " with optional mesh motion and mesh topology changes."
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createDynamicFvMesh.H"
    #include "createDyMControls.H"
    #include "initContinuityErrs.H"
    #include "createFields.H"
    #include "createFieldRefs.H"
    #include "createRhoUfIfPresent.H"

    turbulence->validate();

    if (!LTS)
    {
        #include "compressibleCourantNo.H"
        #include "setInitialDeltaT.H"
    }

    label writeTimes = 0;

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        #include "readDyMControls.H"

        // Store divrhoU from the previous mesh
        // so that it can be mapped and used in correctPhi
        // to ensure the corrected phi has the same divergence
        autoPtr<volScalarField> divrhoU;
        if (correctPhi)
        {
            divrhoU.reset
            (
                new volScalarField
                (
                    "divrhoU",
                    fvc::div(fvc::absolute(phi, rho, U))
                )
            );
        }

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "compressibleCourantNo.H"
            #include "setDeltaT.H"
        }

        ++runTime;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        julia.checkPointers();
        jlc_evolve_cloud(runTime.deltaT().value());

        julia.checkPointers();

        Pout<< "min/max UTransx:    " << min(UTrans.component(0)).value() << "/"
                                      << max(UTrans.component(0)).value() << nl
            << "min/max hTrans:     " << min(hTrans).value() << "/"
                                      << max(hTrans).value() << nl
            << "min/max rhoVTrans:  " << min(rhoVTrans).value() << "/"
                                      << max(rhoVTrans).value() << endl;


        Info<< "--- integral Lagrangian sources ---" << nl
            << "    momentum [kg·m/s]: "
            << fvc::domainIntegrate(UTrans/mesh.V()) << nl
            << "    energy [J]:        "
            << fvc::domainIntegrate(hTrans/mesh.V()) << nl
            << "    H2O mass [kg]:     "
            << fvc::domainIntegrate(rhoVTrans) << endl;        

        dimensionedVector integralMomentum = fvc::domainIntegrate(rho*U);
        dimensionedScalar integralEnergy = fvc::domainIntegrate
        (
            thermo.T()*rho*thermo.Cp()
          + thermo.T()*rhoV*CpH2Og
        );
        dimensionedScalar integralVapourMass = fvc::domainIntegrate(rhoV);

        Info<< "--- initial integral values Euler field ---" << nl
            << "    momentum [kg·m/s]: " << integralMomentum.value() << nl
            << "    energy [J]:        " << integralEnergy.value() << nl
            << "    H2O mass [kg]:     " << integralVapourMass.value() << endl;

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            if (pimple.firstIter() || moveMeshOuterCorrectors)
            {
                // Store momentum to set rhoUf for introduced faces.
                autoPtr<volVectorField> rhoU;
                if (rhoUf.valid())
                {
                    rhoU.reset(new volVectorField("rhoU", rho*U));
                }
            }

            if (pimple.firstIter() && !pimple.SIMPLErho())
            {
                #include "rhoEqn.H"
                #include "rhoVEqn.H"   // transport of water vapour
            }

            #include "UEqn.H"
            #include "EEqn.H"

            // --- Pressure corrector loop
            while (pimple.correct())
            {
                #include "pEqn.H"
            }

            if (pimple.turbCorr())
            {
                turbulence->correct();
            }
        }

        rho = thermo.rho();

        Info<< "T min/max:    " << gMin(thermo.T())<<"/"<<gMax(thermo.T()) << nl
            << "rhoV min/max: " << gMin(rhoV)<< "/" << gMax(rhoV) << endl;

        integralMomentum = fvc::domainIntegrate(rho*U);
        integralEnergy = fvc::domainIntegrate
        (
            thermo.T()*rho*thermo.Cp()
          + thermo.T()*rhoV*CpH2Og
        );
        integralVapourMass = fvc::domainIntegrate(rhoV);

        Info<< "--- integral values Euler field ---" << nl
            << "    momentum [kg·m/s]: " << integralMomentum.value() << nl
            << "    energy [J]:        " << integralEnergy.value() << nl
            << "    H2O mass [kg]:     " << integralVapourMass.value() << endl;

        runTime.write();

        if (runTime.writeTime())
        {
            julia.checkedEvalString
            (
                "if comm.isMaster write(chunks, comm, executor) end"
            );
            ++writeTimes;
        }

        runTime.printExecutionTime(Info);
    }

    if (writeTimes)
    {
        julia.checkedEvalString("write_paraview_collection(comm)");
    }

    Info<< "End\n" << endl;

    // Disentangle OF and Julia memory
    julia.finalize();

    return 0;
}


// ************************************************************************* //
