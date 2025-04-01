/********************************************************************
 * Copyright © 2018 Computational Molecular Biology Group,          *
 *                  Freie Universität Berlin (GER)                  *
 *                                                                  *
 * Redistribution and use in source and binary forms, with or       *
 * without modification, are permitted provided that the            *
 * following conditions are met:                                    *
 *  1. Redistributions of source code must retain the above         *
 *     copyright notice, this list of conditions and the            *
 *     following disclaimer.                                        *
 *  2. Redistributions in binary form must reproduce the above      *
 *     copyright notice, this list of conditions and the following  *
 *     disclaimer in the documentation and/or other materials       *
 *     provided with the distribution.                              *
 *  3. Neither the name of the copyright holder nor the names of    *
 *     its contributors may be used to endorse or promote products  *
 *     derived from this software without specific                  *
 *     prior written permission.                                    *
 *                                                                  *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND           *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,      *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF         *
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE         *
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR            *
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,     *
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,         *
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; *
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER *
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,      *
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)    *
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF      *
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                       *
 ********************************************************************/


/**
 * Declaration of the base class for all order 2 potentials. They basically have calculateForce, calculateEnergy and
 * calculateForceAndEnergy methods, which take a modifiable reference argument and the difference vector x_ij between
 * two particles.
 * Further, subclasses have to implement getCutoffRadius so that the neighbor list can be created more efficiently.
 *
 * @file PotentialOrder2.h
 * @brief Declaration of the base class for all order 2 potentials.
 * @author clonker
 * @date 31.05.16
 */

#pragma once

#include <cmath>

#include "Potential.h"

namespace readdy::model {
class ParticleTypeRegistry;

namespace potentials {
class PotentialRegistry;

class PotentialOrder2 : public Potential {
public:
    PotentialOrder2(ParticleTypeId type1, ParticleTypeId type2)
            : _particleType1(type1), _particleType2(type2) {}

    virtual scalar calculateEnergy(const Vec3 &x_ij) const = 0;

    virtual void calculateForce(Vec3 &force, const Vec3 &x_ij) const = 0;

    void calculateForceAndEnergy(Vec3 &force, scalar &energy, const Vec3 &x_ij) const {
        energy += calculateEnergy(x_ij);
        calculateForce(force, x_ij);
    };

    scalar getCutoffRadius() const {
        return std::sqrt(getCutoffRadiusSquared());
    };

    virtual scalar getCutoffRadiusSquared() const = 0;

    friend std::ostream &operator<<(std::ostream &os, const PotentialOrder2 &potential) {
        os << potential.describe();
        return os;
    }

    ParticleTypeId particleType1() const {
        return _particleType1;
    }

    ParticleTypeId particleType2() const {
        return _particleType2;
    }

protected:
    ParticleTypeId _particleType1, _particleType2;
};

}
}
