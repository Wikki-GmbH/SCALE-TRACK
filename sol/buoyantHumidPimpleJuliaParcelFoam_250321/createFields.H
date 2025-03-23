Info<< "Load Julia program\n" << endl;
jl_eval_string_with_exception("include(\"./asyncParallelTracking.jl\")");

typedef void* (*evolveFunctionType) (const double);
evolveFunctionType jlc_evolve_cloud =
    reinterpret_cast<evolveFunctionType>(get_cfunction_pointer("evolve_cloud_ptr"));
if (!jlc_evolve_cloud) return 1;

juliaWrapper julia;

#include "createRDeltaT.H"

Info<< "Reading thermophysical properties\n" << endl;

autoPtr<rhoThermo> pThermo(rhoThermo::New(mesh));
rhoThermo& thermo = pThermo();
thermo.validate(args.executable(), "h", "e");

Info<< "he-dims: " << thermo.he().dimensions() << endl;

volScalarField rho
(
    IOobject
    (
        "rho",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    thermo.rho()
);

volScalarField& p = thermo.p();

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

#include "compressibleCreatePhi.H"

pressureControl pressureControl(p, rho, pimple.dict(), false);

Info<< "Creating turbulence model\n" << endl;
autoPtr<compressible::turbulenceModel> turbulence
(
    compressible::turbulenceModel::New
    (
        rho,
        U,
        phi,
        thermo
    )
);

#include "readGravitationalAcceleration.H"
#include "readhRef.H"
#include "gh.H"

Info<< "Reading field p_rgh\n" << endl;
volScalarField p_rgh
(
    IOobject
    (
        "p_rgh",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

// Force p_rgh to be consistent with p
p_rgh = p - rho*gh;

mesh.setFluxRequired(p_rgh.name());

label pRefCell = 0;
scalar pRefValue = 0.0;

if (p_rgh.needReference())
{
    setRefCell
    (
        p,
        p_rgh,
        pimple.dict(),
        pRefCell,
        pRefValue
    );

    p += dimensionedScalar
    (
        "p",
        p.dimensions(),
        pRefValue - getRefCellValue(p, pRefCell)
    );
}

dimensionedScalar initialMass("initialMass", fvc::domainIntegrate(rho));

#include "createDpdt.H"

#include "createK.H"

#include "createMRF.H"
#include "createRadiationModel.H"
#include "createFvOptions.H"

const dimensionedScalar rhoMax("rhoMax", dimDensity, GREAT, pimple.dict());
const dimensionedScalar rhoMin("rhoMin", dimDensity, Zero, pimple.dict());

Info<< "Creating momentum source field UTrans\n" << endl;
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
    dimensionedVector(dimMass*dimVelocity, Zero)
);

Info<< "Creating energy source field hTrans\n" << endl;
volScalarField hTrans
(
    IOobject
    (
        "hTrans",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar(dimEnergy, Zero)
);

Info<< "Reading field rhoV\n" << endl; // water vapour density [kg/m³]
volScalarField rhoV
(
    IOobject
    (
        "rhoV",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Creating water vapour source field rhoVTrans\n" << endl;
volScalarField rhoVTrans
(
    IOobject
    (
        "rhoVTrans",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar(dimDensity, Zero)
);

// heat capacity of water vapour
const dimensionedScalar CpH2Og
(
    "CpH2Og",
    dimEnergy/dimMass/dimTemperature,
    1860.0
);

Info<< "Reallocating field U\n" << endl;
julia.reallocate(U);
Info<< "Reallocating field T\n" << endl;
julia.reallocate(thermo.T());
Info<< "Reallocating field rhoV\n" << endl;
julia.reallocate(rhoV);
Info<< "Reallocating field UTrans\n" << endl;
julia.reallocate(UTrans);
Info<< "Reallocating field hTrans\n" << endl;
julia.reallocate(hTrans);
Info<< "Reallocating field rhoVTrans\n" << endl;
julia.reallocate(rhoVTrans);
