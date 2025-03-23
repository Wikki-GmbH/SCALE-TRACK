{
    volScalarField& he = thermo.he();

    auto tSh = tmp<fvScalarMatrix>::New(he, dimEnergy/dimTime);
    auto& Sh = tSh.ref();
    Sh.source() = -hTrans()/(runTime.deltaT());

    fvScalarMatrix EEqn
    (
        fvm::ddt(rho, he) + fvm::div(phi, he)
      + fvc::ddt(rho, K) + fvc::div(phi, K)
      + (
            he.name() == "e"
          ? fvc::div
            (
                fvc::absolute(phi/fvc::interpolate(rho), U),
                p,
                "div(phiv,p)"
            )
          : -dpdt
        )
      - fvm::laplacian(turbulence->alphaEff(), he)
     ==
        rho*(U&g)
      + tSh
      + radiation->Sh(thermo, he)
      + fvOptions(rho, he)
    );

    EEqn.relax();
    fvOptions.constrain(EEqn);
    EEqn.solve();
    fvOptions.correct(he);
    thermo.correct();
    radiation->correct();
    hTrans.field() = 0.0;
}
