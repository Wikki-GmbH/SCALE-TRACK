/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
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

\*---------------------------------------------------------------------------*/

#include "CloudParcel.H"
#include "physicoChemicalConstants.H"

using namespace Foam::constant;

// * * * * * * * * * * *  Protected Member Functions * * * * * * * * * * * * //

template<class ParcelType>
template<class TrackCloudType>
void Foam::CloudParcel<ParcelType>::setCellValues
(
    TrackCloudType& cloud,
    trackingData& td
)
{
    ParcelType::setCellValues(cloud, td);

    tetIndices tetIs = this->currentTetIndices();

    td.Cpc() = td.CpInterp().interpolate(this->coordinates(), tetIs);

    td.Tc() = td.TInterp().interpolate(this->coordinates(), tetIs);

    td.rhoVc() = td.rhoVcInterp().interpolate(this->coordinates(), tetIs);


    if (td.Tc() < cloud.constProps().TMin())
    {
        if (debug)
        {
            WarningInFunction
                << "Limiting observed temperature in cell " << this->cell()
                << " to " << cloud.constProps().TMin() <<  nl << endl;
        }

        td.Tc() = cloud.constProps().TMin();
    }
}


template<class ParcelType>
template<class TrackCloudType>
void Foam::CloudParcel<ParcelType>::cellValueSourceCorrection
(
    TrackCloudType& cloud,
    trackingData& td,
    const scalar dt
)
{
    const label celli = this->cell();
    const scalar massCell = this->massCell(td);

    td.Uc() += cloud.UTrans()[celli]/massCell;

    const scalar CpMean = td.CpInterp().psi()[celli];
    td.Tc() += cloud.hsTrans()[celli]/(CpMean*massCell);

    if (td.Tc() < cloud.constProps().TMin())
    {
        if (debug)
        {
            WarningInFunction
                << "Limiting observed temperature in cell " << celli
                << " to " << cloud.constProps().TMin() <<  nl << endl;
        }

        td.Tc() = cloud.constProps().TMin();
    }
}


template<class ParcelType>
template<class TrackCloudType>
void Foam::CloudParcel<ParcelType>::calcSurfaceValues
(
    TrackCloudType& cloud,
    trackingData& td,
    const scalar T,
    scalar& Ts,
    scalar& rhos,
    scalar& mus,
    scalar& Pr,
    scalar& kappas
) const
{
    // Surface temperature using two thirds rule
    Ts = (2.0*T + td.Tc())/3.0;

    if (Ts < cloud.constProps().TMin())
    {
        if (debug)
        {
            WarningInFunction
                << "Limiting parcel surface temperature to "
                << cloud.constProps().TMin() <<  nl << endl;
        }

        Ts = cloud.constProps().TMin();
    }

    // Assuming thermo props vary linearly with T for small d(T)
    const scalar TRatio = td.Tc()/Ts;

    rhos = td.rhoc()*TRatio;

    tetIndices tetIs = this->currentTetIndices();
    mus = td.muInterp().interpolate(this->coordinates(), tetIs)/TRatio;
    //kappas = td.kappaInterp().interpolate(this->coordinates(), tetIs)/TRatio;
    // use the same kappa as in SCALE-TRACK julia code
    kappas = td.Tc()*8.9182e-5/TRatio;

    Pr = td.Cpc()*mus/kappas;
    Pr = max(ROOTVSMALL, Pr);
}


template<class ParcelType>
template<class TrackCloudType>
void Foam::CloudParcel<ParcelType>::calc
(
    TrackCloudType& cloud,
    trackingData& td,
    const scalar dt
)
{
    // Define local properties at beginning of time step
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    const scalar np0 = this->nParticle_;
    const scalar mass0 = this->mass();


    // Sources
    // ~~~~~~~

    // Explicit momentum source for particle
    vector Su = Zero;

    // Linearised momentum source coefficient
    scalar Spu = 0.0;

    // Momentum transfer from the particle to the carrier phase
    vector dUTrans = Zero;

    // Mass transfer from the particle to the carrier phase
    scalar drhoVTrans = 0.0;

    // Explicit enthalpy source for particle
    scalar Sh = 0.0;

    // Linearised enthalpy source coefficient
    scalar Sph = 0.0;

    // Sensible enthalpy transfer from the particle to the carrier phase
    scalar dhsTrans = 0.0;


    // Mass Transfer
    // Water vapour saturation pressure (Clausius-Clapeyron-Equation) in Pa
    const scalar TcdegC = td.Tc() - 273.15;
    const scalar pvsat =
        611.21*exp((18.678 - TcdegC/234.5)*(TcdegC/(257.14 + TcdegC)));

    // water vapour density at saturation in kg/m³
    // 18.01528e-3 -> molar mass of water
    const scalar rhovsat = 18.01528e-3*pvsat/(physicoChemical::R.value()*td.Tc());

    //air molecular density (43.04*Tref*p/(T*pref)) in mol/m³
    const scalar rhoMolAir = 43.04*283.15/(td.Tc()*1.01325);

    // Water vapour pressure in Pa (assume ambient pressure of 1e5 Pa
    const scalar pv = td.rhoVc()/(rhoMolAir*18.01528e-3)*1.0e5;

    // Saturation ratio of water vapour in continuous phase
    const scalar Sinf = pv/pvsat;

    // Saturation ratio of water vapour at particle surface
    // (Only Kelvin/curvature effect, no Raoult/solute effect)
    const scalar Ssfc = exp
    (
        4.0*18.01528e-3*72.8e-3/
        (physicoChemical::R.value()*this->T_*this->rho_*this->d_)
    );

    // Integration over time (Euler explicit)
    // 24.5e-6 -> diffusivity water in air
    const scalar bcp = 2.0*3.14159*24.5e-6*this->d_*rhovsat;

    // min. particle mass equiv. to ⌀~1µm
    const scalar dm = bcp*(Sinf - Ssfc)*dt;
    const scalar mNew = max(5.0e-16, mass0 + dm);
    drhoVTrans -= mNew - mass0;

    // latent heat release with specific latent heat of vaporisation
    // for water = 2264.71 J/kg
    this->T_ -= drhoVTrans*2264.71/(this->Cp_*0.5*(mNew + mass0));

    this->d_ = cbrt(mNew/this->rho_*6/pi);

    const scalar T0 = this->T_;

    // Calc surface values
    // ~~~~~~~~~~~~~~~~~~~
    scalar Ts, rhos, mus, Pr, kappas;
    calcSurfaceValues(cloud, td, this->T_, Ts, rhos, mus, Pr, kappas);

    // Reynolds number
    scalar Re = this->Re(rhos, this->U_, td.Uc(), this->d_, mus);

    // Heat transfer
    // ~~~~~~~~~~~~~

    // Sum Ni*Cpi*Wi of emission species
    scalar NCpW = 0.0;

    // Calculate new particle temperature
    this->T_ =
        this->calcHeatTransfer
        (
            cloud,
            td,
            dt,
            Re,
            Pr,
            kappas,
            NCpW,
            Sh,
            mass0,
            dhsTrans,
            Sph
        );

    // Motion
    // ~~~~~~

    // Calculate new particle velocity
    this->U_ = this->calcVelocity
        (cloud, td, dt, Re, mus, this->mass(), mass0, Su, dUTrans, Spu);

    //  Accumulate carrier phase source terms
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if (cloud.solution().coupled())
    {
        // Update momentum transfer
        cloud.UTrans()[this->cell()] += np0*dUTrans;

        // Update momentum transfer coefficient
        cloud.UCoeff()[this->cell()] += np0*Spu;

        // Update sensible enthalpy transfer
        cloud.rhoVTrans()[this->cell()] += np0*drhoVTrans;

        // Update sensible enthalpy transfer
        cloud.hsTrans()[this->cell()] += np0*dhsTrans;

        // Update sensible enthalpy coefficient
        cloud.hsCoeff()[this->cell()] += np0*Sph;

        // Update radiation fields
        if (cloud.radiation())
        {
            const scalar ap = this->areaP();
            const scalar T4 = pow4(T0);
            cloud.radAreaP()[this->cell()] += dt*np0*ap;
            cloud.radT4()[this->cell()] += dt*np0*T4;
            cloud.radAreaPT4()[this->cell()] += dt*np0*ap*T4;
        }
    }
}


template<class ParcelType>
template<class TrackCloudType>
Foam::scalar Foam::CloudParcel<ParcelType>::calcHeatTransfer
(
    TrackCloudType& cloud,
    trackingData& td,
    const scalar dt,
    const scalar Re,
    const scalar Pr,
    const scalar kappa,
    const scalar NCpW,
    const scalar Sh,
    const scalar mass0,
    scalar& dhsTrans,
    scalar& Sph
)
{
    if (!cloud.heatTransfer().active())
    {
        return T_;
    }

    const scalar d = this->d();
    const scalar rho = this->rho();
    const scalar As = this->areaS(d);
    const scalar V = this->volume(d);
    const scalar mass = rho*V;

    // Calc heat transfer coefficient
    scalar htc = cloud.heatTransfer().htc(d, Re, Pr, kappa, NCpW);

    // Calculate the integration coefficients
    const scalar bcp = htc*As/(mass*Cp_);
    const scalar acp = bcp*td.Tc();
    scalar ancp = Sh;
    if (cloud.radiation())
    {
        const tetIndices tetIs = this->currentTetIndices();
        const scalar Gc = td.GInterp().interpolate(this->coordinates(), tetIs);
        const scalar sigma = physicoChemical::sigma.value();
        const scalar epsilon = cloud.constProps().epsilon0();

        ancp += As*epsilon*(Gc/4.0 - sigma*pow4(T_));
    }
    ancp /= mass*Cp_;

    // Integrate to find the new parcel temperature
    const scalar deltaT = cloud.TIntegrator().delta(T_, dt, acp + ancp, bcp);
    const scalar deltaTncp = ancp*dt;
    const scalar deltaTcp = deltaT - deltaTncp;

    // Calculate the new temperature and the enthalpy transfer terms
    scalar Tnew = T_ + deltaT;
    Tnew = clamp(Tnew, cloud.constProps().TMin(), cloud.constProps().TMax());

    dhsTrans -= mass*Cp_*deltaTcp;

    Sph = dt*mass*Cp_*bcp;

    return Tnew;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ParcelType>
Foam::CloudParcel<ParcelType>::CloudParcel
(
    const CloudParcel<ParcelType>& p
)
:
    ParcelType(p),
    T_(p.T_),
    Cp_(p.Cp_)
{}


template<class ParcelType>
Foam::CloudParcel<ParcelType>::CloudParcel
(
    const CloudParcel<ParcelType>& p,
    const polyMesh& mesh
)
:
    ParcelType(p, mesh),
    T_(p.T_),
    Cp_(p.Cp_)
{}


// * * * * * * * * * * * * * * IOStream operators  * * * * * * * * * * * * * //

#include "CloudParcelIO.C"

// ************************************************************************* //
