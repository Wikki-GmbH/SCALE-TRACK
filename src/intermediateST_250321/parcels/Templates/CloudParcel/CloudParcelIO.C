/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2016-2022 OpenCFD Ltd.
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
#include "IOstreams.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

template<class ParcelType>
Foam::string Foam::CloudParcel<ParcelType>::propertyList_ =
    Foam::CloudParcel<ParcelType>::propertyList();


template<class ParcelType>
const std::size_t Foam::CloudParcel<ParcelType>::sizeofFields
(
    sizeof(CloudParcel<ParcelType>)
  - offsetof(CloudParcel<ParcelType>, T_)
);


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ParcelType>
Foam::CloudParcel<ParcelType>::CloudParcel
(
    const polyMesh& mesh,
    Istream& is,
    bool readFields,
    bool newFormat
)
:
    ParcelType(mesh, is, readFields, newFormat),
    T_(0.0),
    Cp_(0.0)
{
    if (readFields)
    {
        if (is.format() == IOstreamOption::ASCII)
        {
            is  >> T_ >> Cp_;
        }
        else if (!is.checkLabelSize<>() || !is.checkScalarSize<>())
        {
            // Non-native label or scalar size

            is.beginRawRead();

            readRawScalar(is, &T_);
            readRawScalar(is, &Cp_);

            is.endRawRead();
        }
        else
        {
            is.read(reinterpret_cast<char*>(&T_), sizeofFields);
        }
    }

    is.check(FUNCTION_NAME);
}


template<class ParcelType>
template<class CloudType>
void Foam::CloudParcel<ParcelType>::readFields(CloudType& c)
{
    const bool readOnProc = c.size();

    ParcelType::readFields(c);

    IOField<scalar> T(c.newIOobject("T", IOobject::MUST_READ), readOnProc);
    c.checkFieldIOobject(c, T);

    IOField<scalar> Cp(c.newIOobject("Cp", IOobject::MUST_READ), readOnProc);
    c.checkFieldIOobject(c, Cp);


    label i = 0;
    for (CloudParcel<ParcelType>& p : c)
    {
        p.T_ = T[i];
        p.Cp_ = Cp[i];

        ++i;
    }
}


template<class ParcelType>
template<class CloudType>
void Foam::CloudParcel<ParcelType>::writeFields(const CloudType& c)
{
    ParcelType::writeFields(c);

    const label np = c.size();
    const bool writeOnProc = c.size();

    IOField<scalar> T(c.newIOobject("T", IOobject::NO_READ), np);
    IOField<scalar> Cp(c.newIOobject("Cp", IOobject::NO_READ), np);

    label i = 0;
    for (const CloudParcel<ParcelType>& p : c)
    {
        T[i] = p.T_;
        Cp[i] = p.Cp_;

        ++i;
    }

    T.write(writeOnProc);
    Cp.write(writeOnProc);
}


template<class ParcelType>
void Foam::CloudParcel<ParcelType>::writeProperties
(
    Ostream& os,
    const wordRes& filters,
    const word& delim,
    const bool namesOnly
) const
{
    ParcelType::writeProperties(os, filters, delim, namesOnly);

    #undef  writeProp
    #define writeProp(Name, Value)                                            \
        ParcelType::writeProperty(os, Name, Value, namesOnly, delim, filters)

    writeProp("T", T_);
    writeProp("Cp", Cp_);

    #undef writeProp
}


template<class ParcelType>
template<class CloudType>
void Foam::CloudParcel<ParcelType>::readObjects
(
    CloudType& c,
    const objectRegistry& obr
)
{
    ParcelType::readFields(c);

    if (!c.size()) return;

    auto& T = cloud::lookupIOField<scalar>("T", obr);
    auto& Cp = cloud::lookupIOField<scalar>("Cp", obr);

    label i = 0;
    for (CloudParcel<ParcelType>& p : c)
    {
        p.T_ = T[i];
        p.Cp_ = Cp[i];

        ++i;
    }
}


template<class ParcelType>
template<class CloudType>
void Foam::CloudParcel<ParcelType>::writeObjects
(
    const CloudType& c,
    objectRegistry& obr
)
{
    ParcelType::writeObjects(c, obr);

    const label np = c.size();

    auto& T = cloud::createIOField<scalar>("T", np, obr);
    auto& Cp = cloud::createIOField<scalar>("Cp", np, obr);

    label i = 0;
    for (const CloudParcel<ParcelType>& p : c)
    {
        T[i] = p.T_;
        Cp[i] = p.Cp_;

        ++i;
    }
}


// * * * * * * * * * * * * * * * IOstream Operators  * * * * * * * * * * * * //

template<class ParcelType>
Foam::Ostream& Foam::operator<<
(
    Ostream& os,
    const CloudParcel<ParcelType>& p
)
{
    if (os.format() == IOstreamOption::ASCII)
    {
        os  << static_cast<const ParcelType&>(p)
            << token::SPACE << p.T()
            << token::SPACE << p.Cp();
    }
    else
    {
        os  << static_cast<const ParcelType&>(p);
        os.write
        (
            reinterpret_cast<const char*>(&p.T_),
            CloudParcel<ParcelType>::sizeofFields
        );
    }

    os.check(FUNCTION_NAME);
    return os;
}


// ************************************************************************* //
