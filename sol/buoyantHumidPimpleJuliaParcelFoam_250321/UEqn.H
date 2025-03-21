    // Solve the Momentum equation

    MRF.correctBoundaryVelocity(U);

    auto tSU = tmp<fvVectorMatrix>::New(U, dimForce);
    auto& SU = tSU.ref();
    SU.source() = -UTrans()/(runTime.deltaT());

    fvVectorMatrix UEqn
    (
        fvm::ddt(rho, U) + fvm::div(phi, U)
      + MRF.DDt(rho, U)
      + turbulence->divDevRhoReff(U)
     ==
        tSU
      + fvOptions(rho, U)
    );

    UEqn.relax();

    fvOptions.constrain(UEqn);

    if (pimple.momentumPredictor())
    {
        solve
        (
            UEqn
         ==
            fvc::reconstruct
            (
                (
                  - ghf*fvc::snGrad(rho)
                  - fvc::snGrad(p_rgh)
                )*mesh.magSf()
            )
        );

        fvOptions.correct(U);
        K = 0.5*magSqr(U);
    }

    UTrans.field() = vector::zero;
