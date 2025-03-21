/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2021 OpenCFD Ltd.
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
    buoyantHumidPimpleParcelFoam_250314

Group
    grpHeatTransferSolvers

Description
    Test solver augmented from buoyantPimpleFoam for dilute multiphase flows
    using Euler-Lagrange ansatz. Lagrangian particles use cloudCloud, that is
    OpenFOAM's thermoCloud with additional cloud droplet growth capability.

Author
    Silvio Schmalfuß, TROPOS, 2025

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "rhoThermo.H"
#include "turbulentFluidThermoModel.H"
#include "radiationModel.H"
#include "basicCloudCloud.H"
#include "CorrectPhi.H"
#include "fvOptions.H"
#include "pimpleControl.H"
#include "pressureControl.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
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

    using clock = std::chrono::system_clock;
    using sec = std::chrono::duration<double>;
    sec totalDuration(0.0);

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

        // Store the particle positions
        parcels.storeGlobalPositions();

        auto before = clock::now();
        parcels.evolve();
        sec duration = clock::now() - before;
        totalDuration += duration;

        Info<< "Time spent in parcels.evolve - current: "
            << duration.count() << "s" << nl
            << "Time spent in parcels.evolve - total:   "
            << totalDuration.count() << "s" << endl;

        Info<< "--- integral Lagrangian sources ---" << nl
            << "    momentum [kg·m/s]: "
            << fvc::domainIntegrate(parcels.UTrans()/mesh.V()) << nl
            << "    energy [J]:        "
            << fvc::domainIntegrate(parcels.hsTrans()/mesh.V()) << nl
            << "    H2O mass [kg]:     "
            << fvc::domainIntegrate(parcels.rhoVTrans()) << endl;        

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

                // Do any mesh changes
                mesh.update();

                if (mesh.changing())
                {
                    gh = (g & mesh.C()) - ghRef;
                    ghf = (g & mesh.Cf()) - ghRef;

                    MRF.update();

                    if (correctPhi)
                    {
                        // Calculate absolute flux
                        // from the mapped surface velocity
                        phi = mesh.Sf() & rhoUf();

                        #include "correctPhi.H"

                        // Make the fluxes relative to the mesh-motion
                        fvc::makeRelative(phi, rho, U);
                    }

                    if (checkMeshCourantNo)
                    {
                        #include "meshCourantNo.H"
                    }
                }
            }

            if (pimple.firstIter() && !pimple.SIMPLErho())
            {
                #include "rhoEqn.H"
            }

            #include "UEqn.H"
            #include "EEqn.H"
            #include "rhoVEqn.H"   // transport of water vapour

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

        Info<< "--- final integral values Euler field ---" << nl
            << "    momentum [kg·m/s]: " << integralMomentum.value() << nl
            << "    energy [J]:        " << integralEnergy.value() << nl
            << "    H2O mass [kg]:     " << integralVapourMass.value() << endl;

        runTime.write();

        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
