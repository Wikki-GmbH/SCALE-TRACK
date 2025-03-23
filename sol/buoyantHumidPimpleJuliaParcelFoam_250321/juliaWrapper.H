/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2024 Sergey Lesnik
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

Class
    Foam::juliaWrapper

Description
    Wrapper for the interface functions to Julia

Author
    Sergey Lesnik, Wikki GmbH, 2024

\*---------------------------------------------------------------------------*/

#ifndef Foam_juliaWrapper_H
#define Foam_juliaWrapper_H

#include <julia.h>

extern "C"
{
#   include "juliaHelper.h"
}

#include "GeometricField.H"
#include "volFields.H"

namespace Foam
{

/*---------------------------------------------------------------------------*\
                          Class juliaWrapper Declaration
\*---------------------------------------------------------------------------*/

class juliaWrapper
{
    // Private Typedefs

        typedef GeometricField<scalar, fvPatchField, volMesh> sGeoField;
        typedef GeometricField<vector, fvPatchField, volMesh> vGeoField;


    // Private Data

        //- Table with pointer to data allocated by Julia and to the associated
        //  scalar geometric field
        DynamicList<Tuple2<scalar*, sGeoField*>> scalarPointerTable_;

        //- Table with pointer to data allocated by Julia and to the associated
        //  vector geometric field
        DynamicList<Tuple2<vector*, vGeoField*>> vectorPointerTable_;


    // Private Member Functions

        template<class Type>
        void pointersOk
        (
            const DynamicList
            <
                Tuple2<Type*, GeometricField<Type, fvPatchField, volMesh>*>
            >& ptrList
        )
        {
            for (auto entry : ptrList)
            {
                if (entry.first() != entry.second()->internalFieldRef().data())
                {
                    FatalErrorInFunction
                        << "Pointer to data of field " << entry.second()->name()
                        << " has changed since the allocation by Julia"
                        << exit(FatalError);
                }
            }
        }


public:

    // Member Functions

    template<class Type>
    void reallocate
    (
        const GeometricField<Type, fvPatchField, volMesh>& cField
    )
    {
        GeometricField<Type, fvPatchField, volMesh>& field =
            const_cast<GeometricField<Type, fvPatchField, volMesh>&>
            (
                cField
            );

        reallocate(field);
    }

    template<class Type>
    void reallocate
    (
        GeometricField<Type, fvPatchField, volMesh>& field
    )
    {
        label nCmpts = pTraits_nComponents<Type>::value;
        typedef typename pTraits_cmptType<Type>::type cmptType;
        label cmptSize = sizeof(cmptType);

        typedef Type* (*fType) (const char*, int, int, int);
        fType allocate_array =
            reinterpret_cast<fType>(get_cfunction_pointer("allocate_array_ptr"));

        // Allocate array by Julia and get its pointer
        Type* dataPtr =
            allocate_array
            (
                field.name().c_str(),
                field.size(),
                nCmpts,
                sizeof(cmptSize)
            );

        // Store pointer
        if constexpr (std::is_same<Type, scalar>::value)
        {
            scalarPointerTable_.push_back
            (
                Tuple2<scalar*, sGeoField*>(dataPtr, &field)
            );
        }
        else if constexpr (std::is_same<Type, vector>::value)
        {
            vectorPointerTable_.push_back
            (
                Tuple2<vector*, vGeoField*>(dataPtr, &field)
            );
        }
        else
        {
            FatalErrorInFunction
                << "Reallocating of fields with type "
                << pTraits<Type>::typeName
                << " is not supported" << exit(FatalError);
        }

        UList<Type> ul(dataPtr, field.size());

        // Copy the data from the OpenFOAM's field to the newly allocated one
        // and swap the pointers
        ul.deepCopy(field.internalField());
        field.internalFieldRef().swap(ul);
    }

    void finalize();

    void checkPointers();

    void checkedEvalString(const char* str);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
