/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    SPDX-License-Identifier: GPL-3.0-or-later

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
#include <csignal>

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
    int jl_argc = 3;
    char** jl_argv = static_cast<char**>(malloc(sizeof(char*)*jl_argc));
    jl_argv[0] = argv[0];
    // 1,1: an interactive thread for OpenFOAM and a default thread for Julia.
    // jl_parse_opts ignores the JULIA_NUM_THREADS environment variable, so
    // forward it here to enable multi-threaded particle tracking on the CPU.
    string threadsOpt = "--threads=1,1";
    const string nJlThreads(getEnv("JULIA_NUM_THREADS"));
    if (!nJlThreads.empty())
    {
        threadsOpt = "--threads=" + nJlThreads + ",1";
    }
    // "@." walks up from the working directory (the case directory) until it
    // finds a Project.toml, so a case picks up the shared environment at the
    // repository root, or its own Project.toml if it carries one.
    // JULIA_PROJECT wins, for running against an environment elsewhere.
    const string jlProject(getEnv("JULIA_PROJECT"));
    string projectOpt =
        "--project=" + (jlProject.empty() ? string("@.") : jlProject);
    jl_argv[1] = const_cast<char*>(threadsOpt.c_str());
    jl_argv[2] = const_cast<char*>(projectOpt.c_str());
    jl_parse_opts(&jl_argc, &jl_argv);
    jl_init();
    Info<< "Initialising Julia - done" << endl;

    // Julia's multi-threaded garbage collector stops the world by making the
    // running threads fault on a protected safepoint page, which its SIGSEGV
    // handler recognizes and parks the thread.  Save the handler here:
    // OpenFOAM's argument parsing and the MPI library install their own
    // handlers later, which would turn every safepoint hit into a fatal
    // "segmentation fault".
    struct sigaction juliaSegvAction;
    sigaction(SIGSEGV, nullptr, &juliaSegvAction);

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

    // argList has installed OpenFOAM's handlers; restore Julia's SIGSEGV
    // handler before the tracking script is loaded and spawns the Julia tasks.
    sigaction(SIGSEGV, &juliaSegvAction, nullptr);

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
            // Assignment would reallocate the field's storage, which the
            // zero-copy sharing with the tracking forbids -- copy into it
            U.deepCopy(HbyA - rAU*fvc::grad(p));
            U.correctBoundaryConditions();
        }

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
