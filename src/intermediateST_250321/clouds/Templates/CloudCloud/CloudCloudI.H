/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2023 OpenCFD Ltd.
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

#include "physicoChemicalConstants.H"

using namespace Foam::constant;

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class CloudType>
inline const Foam::CloudCloud<CloudType>&
Foam::CloudCloud<CloudType>::cloudCopy() const
{
    return *cloudCopyPtr_;
}


template<class CloudType>
inline const typename CloudType::particleType::constantProperties&
Foam::CloudCloud<CloudType>::constProps() const
{
    return constProps_;
}


template<class CloudType>
inline typename CloudType::particleType::constantProperties&
Foam::CloudCloud<CloudType>::constProps()
{
    return constProps_;
}


template<class CloudType>
inline const Foam::SLGThermo& Foam::CloudCloud<CloudType>::thermo() const
{
    return thermo_;
}


template<class CloudType>
inline const Foam::volScalarField& Foam::CloudCloud<CloudType>::T() const
{
    return T_;
}


template<class CloudType>
inline const Foam::volScalarField& Foam::CloudCloud<CloudType>::p() const
{
    return p_;
}


template<class CloudType>
inline const Foam::volScalarField& Foam::CloudCloud<CloudType>::rhoV() const
{
    return rhoV_;
}


template<class CloudType>
inline const Foam::HeatTransferModel<Foam::CloudCloud<CloudType>>&
Foam::CloudCloud<CloudType>::heatTransfer() const
{
    return *heatTransferModel_;
}


template<class CloudType>
inline const Foam::integrationScheme&
Foam::CloudCloud<CloudType>::TIntegrator() const
{
    return *TIntegrator_;
}


template<class CloudType>
inline bool Foam::CloudCloud<CloudType>::radiation() const
{
    return radiation_;
}


template<class CloudType>
inline Foam::DimensionedField<Foam::scalar, Foam::volMesh>&
Foam::CloudCloud<CloudType>::radAreaP()
{
    if (!radiation_)
    {
        FatalErrorInFunction
            << "Radiation field requested, but radiation model not active"
            << abort(FatalError);
    }

    return *radAreaP_;
}


template<class CloudType>
inline const Foam::DimensionedField<Foam::scalar, Foam::volMesh>&
Foam::CloudCloud<CloudType>::radAreaP() const
{
    if (!radiation_)
    {
        FatalErrorInFunction
            << "Radiation field requested, but radiation model not active"
            << abort(FatalError);
    }

    return *radAreaP_;
}


template<class CloudType>
inline Foam::DimensionedField<Foam::scalar, Foam::volMesh>&
Foam::CloudCloud<CloudType>::radT4()
{
    if (!radiation_)
    {
        FatalErrorInFunction
            << "Radiation field requested, but radiation model not active"
            << abort(FatalError);
    }

    return *radT4_;
}


template<class CloudType>
inline const Foam::DimensionedField<Foam::scalar, Foam::volMesh>&
Foam::CloudCloud<CloudType>::radT4() const
{
    if (!radiation_)
    {
        FatalErrorInFunction
            << "Radiation field requested, but radiation model not active"
            << abort(FatalError);
    }

    return *radT4_;
}


template<class CloudType>
inline Foam::DimensionedField<Foam::scalar, Foam::volMesh>&
Foam::CloudCloud<CloudType>::radAreaPT4()
{
    if (!radiation_)
    {
        FatalErrorInFunction
            << "Radiation field requested, but radiation model not active"
            << abort(FatalError);
    }

    return *radAreaPT4_;
}


template<class CloudType>
inline const Foam::DimensionedField<Foam::scalar, Foam::volMesh>&
Foam::CloudCloud<CloudType>::radAreaPT4() const
{
    if (!radiation_)
    {
        FatalErrorInFunction
            << "Radiation field requested, but radiation model not active"
            << abort(FatalError);
    }

    return *radAreaPT4_;
}


template<class CloudType>
inline void Foam::CloudCloud<CloudType>::transferToCarrier
(
    const parcelType& p,
    const typename parcelType::trackingData& td
)
{
    CloudType::transferToCarrier(p, td);

    hsTrans()[p.cell()] += p.nParticle()*p.mass()*p.hs();
}


template<class CloudType>
inline Foam::DimensionedField<Foam::scalar, Foam::volMesh>&
Foam::CloudCloud<CloudType>::rhoVTrans()
{
    return *rhoVTrans_;
}


template<class CloudType>
inline const Foam::DimensionedField<Foam::scalar, Foam::volMesh>&
Foam::CloudCloud<CloudType>::rhoVTrans() const
{
    return *rhoVTrans_;
}


template<class CloudType>
inline Foam::DimensionedField<Foam::scalar, Foam::volMesh>&
Foam::CloudCloud<CloudType>::hsTrans()
{
    return *hsTrans_;
}


template<class CloudType>
inline const Foam::DimensionedField<Foam::scalar, Foam::volMesh>&
Foam::CloudCloud<CloudType>::hsTrans() const
{
    return *hsTrans_;
}


template<class CloudType>
inline Foam::DimensionedField<Foam::scalar, Foam::volMesh>&
Foam::CloudCloud<CloudType>::hsCoeff()
{
    return *hsCoeff_;
}


template<class CloudType>
inline const Foam::DimensionedField<Foam::scalar, Foam::volMesh>&
Foam::CloudCloud<CloudType>::hsCoeff() const
{
    return *hsCoeff_;
}


template<class CloudType>
inline Foam::tmp<Foam::fvScalarMatrix>
Foam::CloudCloud<CloudType>::Sh(volScalarField& hs) const
{
    DebugInfo
        << "hsTrans min/max = " << min(hsTrans()).value() << ", "
        << max(hsTrans()).value() << nl
        << "hsCoeff min/max = " << min(hsCoeff()).value() << ", "
        << max(hsCoeff()).value() << endl;

    if (this->solution().coupled())
    {
        if (this->solution().semiImplicit("h"))
        {
            const volScalarField Cp(thermo_.thermo().Cp());
            const volScalarField::Internal
                Vdt(this->mesh().V()*this->db().time().deltaT());

            return
                hsTrans()/Vdt
              - fvm::SuSp(hsCoeff()/(Cp*Vdt), hs)
              + hsCoeff()/(Cp*Vdt)*hs;
        }
        else
        {
            auto tfvm = tmp<fvScalarMatrix>::New(hs, dimEnergy/dimTime);
            auto& fvm = tfvm.ref();

            fvm.source() = -hsTrans()/(this->db().time().deltaT());

            return tfvm;
        }
    }

    return tmp<fvScalarMatrix>::New(hs, dimEnergy/dimTime);
}


template<class CloudType>
inline Foam::tmp<Foam::fvScalarMatrix>
Foam::CloudCloud<CloudType>::SrhoV(volScalarField& rhoV) const
{
    if (this->solution().coupled())
    {
        auto trhoVTrans = volScalarField::New
        (
            IOobject::scopedName(this->name(), "rhoVTrans"),
            IOobject::NO_REGISTER,
            this->mesh(),
            dimensionedScalar(dimMass/dimTime/dimVolume, Zero)
        );
        scalarField& sourceField = trhoVTrans.ref();

        if (this->solution().semiImplicit("rhoV"))
        {
            sourceField = rhoVTrans_;
            sourceField /= this->db().time().deltaTValue()*this->mesh().V();
            return fvm::SuSp(trhoVTrans()/rhoV, rhoV);
        }
        else
        {
            auto tfvm = tmp<fvScalarMatrix>::New(rhoV, dimMass/dimTime);
            auto& fvm = tfvm.ref();
            sourceField = rhoVTrans_;
            fvm.source() = -trhoVTrans()/this->db().time().deltaT();
            return tfvm;
        }
    }

    return tmp<fvScalarMatrix>::New(rhoV, dimMass/dimTime);
}


template<class CloudType>
inline Foam::tmp<Foam::volScalarField> Foam::CloudCloud<CloudType>::Ep() const
{
    auto tEp = tmp<volScalarField>::New
    (
        this->newIOobject
        (
            IOobject::scopedName(this->name(), "radiation", "Ep")
        ),
        this->mesh(),
        dimensionedScalar(dimMass/dimLength/pow3(dimTime), Zero)
    );

    if (radiation_)
    {
        scalarField& Ep = tEp.ref().primitiveFieldRef();
        const scalar dt = this->db().time().deltaTValue();
        const scalarField& V = this->mesh().V();
        const scalar epsilon = constProps_.epsilon0();
        const scalarField& sumAreaPT4 = radAreaPT4_->field();

        Ep = sumAreaPT4*epsilon*physicoChemical::sigma.value()/V/dt;
    }

    return tEp;
}


template<class CloudType>
inline Foam::tmp<Foam::volScalarField> Foam::CloudCloud<CloudType>::ap() const
{
    auto tap = tmp<volScalarField>::New
    (
        this->newIOobject
        (
            IOobject::scopedName(this->name(), "radiation", "ap")
        ),
        this->mesh(),
        dimensionedScalar(dimless/dimLength, Zero)
    );

    if (radiation_)
    {
        scalarField& ap = tap.ref().primitiveFieldRef();
        const scalar dt = this->db().time().deltaTValue();
        const scalarField& V = this->mesh().V();
        const scalar epsilon = constProps_.epsilon0();
        const scalarField& sumAreaP = radAreaP_->field();

        ap = sumAreaP*epsilon/V/dt;
    }

    return tap;
}


template<class CloudType>
inline Foam::tmp<Foam::volScalarField>
Foam::CloudCloud<CloudType>::sigmap() const
{
    auto tsigmap = tmp<volScalarField>::New
    (
        this->newIOobject
        (
            IOobject::scopedName(this->name(), "radiation", "sigmap")
        ),
        this->mesh(),
        dimensionedScalar(dimless/dimLength, Zero)
    );

    if (radiation_)
    {
        scalarField& sigmap = tsigmap.ref().primitiveFieldRef();
        const scalar dt = this->db().time().deltaTValue();
        const scalarField& V = this->mesh().V();
        const scalar epsilon = constProps_.epsilon0();
        const scalar f = constProps_.f0();
        const scalarField& sumAreaP = radAreaP_->field();

        sigmap = sumAreaP*(1.0 - f)*(1.0 - epsilon)/V/dt;
    }

    return tsigmap;
}


template<class CloudType>
inline Foam::scalar Foam::CloudCloud<CloudType>::Tmax() const
{
    scalar val = -GREAT;
    bool nonEmpty = false;

    for (const parcelType& p : *this)
    {
        val = max(val, p.T());
        nonEmpty = true;
    }

    if (returnReduceOr(nonEmpty))
    {
        return returnReduce(val, maxOp<scalar>());
    }

    return 0;
}


template<class CloudType>
inline Foam::scalar Foam::CloudCloud<CloudType>::Tmin() const
{
    scalar val = GREAT;
    bool nonEmpty = false;

    for (const parcelType& p : *this)
    {
        val = min(val, p.T());
        nonEmpty = true;
    }

    if (returnReduceOr(nonEmpty))
    {
        return returnReduce(val, minOp<scalar>());
    }

    return 0;
}


template<class CloudType>
inline Foam::scalar
Foam::CloudCloud<CloudType>::thermalEnergyOfSystem() const
{
    scalar thermalEnergy = 0;

    for (const parcelType& p : *this)
    {
        thermalEnergy += p.nParticle()*p.mass()*p.T()*p.Cp();
    }

    return thermalEnergy;
}



// ************************************************************************* //
