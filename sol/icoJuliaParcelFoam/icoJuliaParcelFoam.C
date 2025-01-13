/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
    Copyright (C) 2024 Sergey Lesnik
    Copyright (C) 2024 Henrik Rusche
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
    icoJuliaParcelFoam

Description
    Test solver augmented from icoFoam for dilute multiphase flows using
    coupling between OpenFOAM and Julia code using Euler-Lagrange ansatz.
    OpenFOAM is responsible for the carrier and Julia for the dispersed phase.
    The Julia file is located in the case directory.

Author
    Sergey Lesnik, Wikki GmbH, 2024
    Henrik Rusche, Wikki GmbH, 2024

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "pisoControl.H"
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
        "Transient solver for incompressible, laminar flow"
        " of Newtonian fluids."
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    pisoControl piso(mesh);

    #include "createFields.H"
    #include "initContinuityErrs.H"

    label writeTimes = 0;

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        julia.checkPointers();
        jlc_evolve_cloud(runTime.deltaT().value());

        #include "CourantNo.H"

        // Momentum predictor

        fvVectorMatrix UEqn
        (
            fvm::ddt(U)
          + fvm::div(phi, U)
          - fvm::laplacian(nu, U)
          ==
            UTrans/runTime.deltaT()
        );

        if (piso.momentumPredictor())
        {
            solve(UEqn == -fvc::grad(p));
        }

        // --- PISO loop
        while (piso.correct())
        {
            volScalarField rAU(1.0/UEqn.A());
            volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
            surfaceScalarField phiHbyA
            (
                "phiHbyA",
                fvc::flux(HbyA)
              + fvc::interpolate(rAU)*fvc::ddtCorr(U, phi)
            );

            adjustPhi(phiHbyA, U, p);

            // Update the pressure BCs to ensure flux consistency
            constrainPressure(p, U, phiHbyA, rAU);

            // Non-orthogonal pressure corrector loop
            while (piso.correctNonOrthogonal())
            {
                // Pressure corrector

                fvScalarMatrix pEqn
                (
                    fvm::laplacian(rAU, p) == fvc::div(phiHbyA)
                );

                pEqn.setReference(pRefCell, pRefValue);

                pEqn.solve(p.select(piso.finalInnerIter()));

                if (piso.finalNonOrthogonalIter())
                {
                    phi = phiHbyA - pEqn.flux();
                }
            }

            #include "continuityErrs.H"
            // Assignment to tmp leads to data reallocation, use deepCopy
            // instead
            // U = HbyA - rAU*fvc::grad(p);
            U.deepCopy(HbyA - rAU*fvc::grad(p));
            U.correctBoundaryConditions();
        }

        runTime.write();

        if (runTime.writeTime())
        {
            julia.checkedEvalString("write(chunk, comm, executor)");
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
