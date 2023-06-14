subroutine ci_func(ci, fval, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z,&
    rh_can, gs_mol, atm2lnd_inst, photosyns_inst)
!
!! DESCRIPTION:
! evaluate the function
! f(ci)=ci - (ca - (1.37rb+1.65rs))*patm*an
!
! remark:  I am attempting to maintain the original code structure, also
! considering one may be interested to output relevant variables for the
! photosynthesis model, I have decided to add these relevant variables to
! the relevant data types.
!
!!ARGUMENTS:
real(r8)             , intent(in)    :: ci       ! intracellular leaf CO2 (Pa)
real(r8)             , intent(in)    :: lmr_z    ! canopy layer: leaf maintenance respiration rate (umol CO2/m**2/s)
real(r8)             , intent(in)    :: par_z    ! par absorbed per unit lai for canopy layer (w/m**2)
real(r8)             , intent(in)    :: gb_mol   ! leaf boundary layer conductance (umol H2O/m**2/s)
real(r8)             , intent(in)    :: je       ! electron transport rate (umol electrons/m**2/s)
real(r8)             , intent(in)    :: cair     ! Atmospheric CO2 partial pressure (Pa)
real(r8)             , intent(in)    :: oair     ! Atmospheric O2 partial pressure (Pa)
real(r8)             , intent(in)    :: rh_can   ! canopy air realtive humidity
integer              , intent(in)    :: p, iv, c ! pft, vegetation type and column indexes
real(r8)             , intent(out)   :: fval     ! return function of the value f(ci)
real(r8)             , intent(out)   :: gs_mol   ! leaf stomatal conductance (umol H2O/m**2/s)
type(atm2lnd_type)   , intent(in)    :: atm2lnd_inst
type(photosyns_type) , intent(inout) :: photosyns_inst
!
!local variables
real(r8) :: ai                  ! intermediate co-limited photosynthesis (umol CO2/m**2/s)
real(r8) :: cs                  ! CO2 partial pressure at leaf surface (Pa)
real(r8) :: term                 ! intermediate in Medlyn stomatal model
real(r8) :: aquad, bquad, cquad  ! terms for quadratic equations
real(r8) :: r1, r2               ! roots of quadratic equation
!------------------------------------------------------------------------------

associate(&
        forc_pbot  => atm2lnd_inst%forc_pbot_downscaled_col   , & ! Output: [real(r8) (:)   ]  atmospheric pressure (Pa)
        c3flag     => photosyns_inst%c3flag_patch             , & ! Output: [logical  (:)   ]  true if C3 and false if C4
        ivt        => patch%itype                             , & ! Input:  [integer  (:)   ]  patch vegetation type
        medlynslope      => pftcon%medlynslope                , & ! Input:  [real(r8) (:)   ]  Slope for Medlyn stomatal conductance model method
        medlynintercept  => pftcon%medlynintercept            , & ! Input:  [real(r8) (:)   ]  Intercept for Medlyn stomatal conductance model method
        stomatalcond_mtd => photosyns_inst%stomatalcond_mtd   , & ! Input:  [integer        ]  method type to use for stomatal conductance
        ac         => photosyns_inst%ac_patch                 , & ! Output: [real(r8) (:,:) ]  Rubisco-limited gross photosynthesis (umol CO2/m**2/s)
        aj         => photosyns_inst%aj_patch                 , & ! Output: [real(r8) (:,:) ]  RuBP-limited gross photosynthesis (umol CO2/m**2/s)
        ap         => photosyns_inst%ap_patch                 , & ! Output: [real(r8) (:,:) ]  product-limited (C3) or CO2-limited (C4) gross photosynthesis (umol CO2/m**2/s)
        ag         => photosyns_inst%ag_patch                 , & ! Output: [real(r8) (:,:) ]  co-limited gross leaf photosynthesis (umol CO2/m**2/s)
        an         => photosyns_inst%an_patch                 , & ! Output: [real(r8) (:,:) ]  net leaf photosynthesis (umol CO2/m**2/s)
        vcmax_z    => photosyns_inst%vcmax_z_patch            , & ! Input:  [real(r8) (:,:) ]  maximum rate of carboxylation (umol co2/m**2/s)
        cp         => photosyns_inst%cp_patch                 , & ! Output: [real(r8) (:)   ]  CO2 compensation point (Pa)
        kc         => photosyns_inst%kc_patch                 , & ! Output: [real(r8) (:)   ]  Michaelis-Menten constant for CO2 (Pa)
        ko         => photosyns_inst%ko_patch                 , & ! Output: [real(r8) (:)   ]  Michaelis-Menten constant for O2 (Pa)
        qe         => photosyns_inst%qe_patch                 , & ! Output: [real(r8) (:)   ]  quantum efficiency, used only for C4 (mol CO2 / mol photons)
        tpu_z      => photosyns_inst%tpu_z_patch              , & ! Output: [real(r8) (:,:) ]  triose phosphate utilization rate (umol CO2/m**2/s)
        kp_z       => photosyns_inst%kp_z_patch               , & ! Output: [real(r8) (:,:) ]  initial slope of CO2 response curve (C4 plants)
        bbb        => photosyns_inst%bbb_patch                , & ! Output: [real(r8) (:)   ]  Ball-Berry minimum leaf conductance (umol H2O/m**2/s)
        mbb        => photosyns_inst%mbb_patch                  & ! Output: [real(r8) (:)   ]  Ball-Berry slope of conductance-photosynthesis relationship
        )

    if (c3flag(p)) then
        ! C3: Rubisco-limited photosynthesis
        ac(p,iv) = vcmax_z(p,iv) * max(ci-cp(p), 0._r8) / (ci+kc(p)*(1._r8+oair/ko(p)))

        ! C3: RuBP-limited photosynthesis
        aj(p,iv) = je * max(ci-cp(p), 0._r8) / (4._r8*ci+8._r8*cp(p))

        ! C3: Product-limited photosynthesis
        ap(p,iv) = 3._r8 * tpu_z(p,iv)

    else

        ! C4: Rubisco-limited photosynthesis
        ac(p,iv) = vcmax_z(p,iv)

        ! C4: RuBP-limited photosynthesis
        aj(p,iv) = qe(p) * par_z * 4.6_r8

        ! C4: PEP carboxylase-limited (CO2-limited)
        ap(p,iv) = kp_z(p,iv) * max(ci, 0._r8) / forc_pbot(c)

    end if

    ! Gross photosynthesis. First co-limit ac and aj. Then co-limit ap

    aquad = params_inst%theta_cj(ivt(p))
    bquad = -(ac(p,iv) + aj(p,iv))
    cquad = ac(p,iv) * aj(p,iv)
    call quadratic (aquad, bquad, cquad, r1, r2)
    ai = min(r1,r2)

    aquad = params_inst%theta_ip
    bquad = -(ai + ap(p,iv))
    cquad = ai * ap(p,iv)
    call quadratic (aquad, bquad, cquad, r1, r2)
    ag(p,iv) = max(0._r8,min(r1,r2))

    ! Net photosynthesis. Exit iteration if an < 0

    an(p,iv) = ag(p,iv) - lmr_z
    if (an(p,iv) < 0._r8) then
        fval = 0._r8
        return
    endif
    ! Quadratic gs_mol calculation with an known. Valid for an >= 0.
    ! With an <= 0, then gs_mol = bbb or medlyn intercept
    cs = cair - 1.4_r8/gb_mol * an(p,iv) * forc_pbot(c)
    cs = max(cs,max_cs)
    if ( stomatalcond_mtd == stomatalcond_mtd_medlyn2011 )then
        term = 1.6_r8 * an(p,iv) / (cs / forc_pbot(c) * 1.e06_r8)
        aquad = 1.0_r8
        bquad = -(2.0 * (medlynintercept(patch%itype(p))*1.e-06_r8 + term) + (medlynslope(patch%itype(p)) * term)**2 / &
            (gb_mol*1.e-06_r8 * rh_can))
        cquad = medlynintercept(patch%itype(p))*medlynintercept(patch%itype(p))*1.e-12_r8 + &
            (2.0*medlynintercept(patch%itype(p))*1.e-06_r8 + term * &
            (1.0 - medlynslope(patch%itype(p))* medlynslope(patch%itype(p)) / rh_can)) * term

        call quadratic (aquad, bquad, cquad, r1, r2)
        gs_mol = max(r1,r2) * 1.e06_r8
    else if ( stomatalcond_mtd == stomatalcond_mtd_bb1987 )then
        aquad = cs
        bquad = cs*(gb_mol - bbb(p)) - mbb(p)*an(p,iv)*forc_pbot(c)
        cquad = -gb_mol*(cs*bbb(p) + mbb(p)*an(p,iv)*forc_pbot(c)*rh_can)
        call quadratic (aquad, bquad, cquad, r1, r2)
        gs_mol = max(r1,r2)
    end if




    ! Derive new estimate for ci

    fval =ci - cair + an(p,iv) * forc_pbot(c) * (1.4_r8*gs_mol+1.6_r8*gb_mol) / (gb_mol*gs_mol)

end associate

end subroutine ci_func