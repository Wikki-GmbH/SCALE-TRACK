Info<< "Reading transportProperties\n" << endl;

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);

dimensionedScalar nu
(
    "nu",
    dimViscosity,
    transportProperties
);

Info<< "Load Julia program\n" << endl;
jl_eval_string_with_exception("include(\"./asyncParallelTracking.jl\")");

typedef void* (*evolveFunctionType) (const double);
evolveFunctionType jlc_evolve_cloud =
    reinterpret_cast<evolveFunctionType>(get_cfunction_pointer("evolve_cloud_ptr"));
if (!jlc_evolve_cloud) return 1;

Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

juliaWrapper julia;
julia.reallocate(U);

Info<< "Creating field momentum source UTrans\n" << endl;
volVectorField UTrans
(
    IOobject
    (
        "UTrans",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector(dimVelocity, Zero)
);

julia.reallocate(UTrans);

#include "createPhi.H"


label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, mesh.solutionDict().subDict("PISO"), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());
