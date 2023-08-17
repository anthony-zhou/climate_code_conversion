subroutine ci_func(ci, fval, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z, rh_can, gs_mol)
    !
    !! DESCRIPTION:
    ! evaluate the function
    ! f(ci)=ci - (ca - (1.37rb+1.65rs))*patm*an
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
    !type(atm2lnd_type)   , intent(in)    :: atm2lnd_inst
    !type(photosyns_type) , intent(inout) :: photosyns_inst
    !
    !local variables
    real(r8) :: ai                  ! intermediate co-limited photosynthesis (umol CO2/m**2/s)
    real(r8) :: cs                  ! CO2 partial pressure at leaf surface (Pa)
    real(r8) :: term                 ! intermediate in Medlyn stomatal model
    real(r8) :: aquad, bquad, cquad  ! terms for quadratic equations
    real(r8) :: r1, r2               ! roots of quadratic equation
    !------------------------------------------------------------------------------
    ! LRH CHANGES FOR UNIT TEST
    
    real(r8) :: ac, aj, ap ! gross photosynthesis (umol CO2/m**2/s)
    real(r8) :: ag, an, es
    
    
   real(r8) :: bbb, cp, forc_pbot, ko, kc, kp_z, mbb, qe, tpu_z, vcmax_z
   logical :: c3flag
   real(r8) :: medlynintercept, medlynslope, theta_cj, theta_ip
   integer :: stomatalcond_mtd, stomatalcond_mtd_medlyn2011, stomatalcond_mtd_bb1987

   forc_pbot = 121000._r8 ! atmospheric pressure (Pa)
   c3flag    = .true. ! true if C3 and false if C4
   medlynslope =  6._r8! Slope for Medlyn stomatal conductance model method
   medlynintercept = 100._r8 ! Intercept for Medlyn stomatal conductance model method
   stomatalcond_mtd = 1 ! method type to use for stomatal conductance (Medlyn or Ball-Berry)
   vcmax_z =  62.5_r8 ! maximum rate of carboxylation (umol co2/m**2/s)
   cp =  4.275_r8 ! CO2 compensation point (Pa)
   kc =  40.49_r8 ! Michaelis-Menten constant for CO2 (Pa)
   ko =  27840._r8 ! Michaelis-Menten constant for O2 (Pa)
   qe =  1.0_r8 ! place holder ! quantum efficiency, used only for C4 (mol CO2 / mol photons)
   tpu_z =  31.5_r8 ! triose phosphate utilization rate (umol CO2/m**2/s)
   kp_z = 1.0_r8 ! place holder ! initial slope of CO2 response curve (C4 plants)
   bbb =  100._r8 ! Ball-Berry minimum leaf conductance (umol H2O/m**2/s)
   mbb =  9._r8 ! Ball-Berry slope of conductance-photosynthesis relationship
   theta_cj = 0.98_r8 !
   theta_ip = 0.95_r8 !
   stomatalcond_mtd_medlyn2011 = 1
   stomatalcond_mtd_bb1987 = 2
    
    ! END LRH CHANGES FOR UNIT TEST
    !------------------------------------------------------------------------------

      if (c3flag) then
         ! C3: Rubisco-limited photosynthesis
         ac = vcmax_z * max(ci-cp, 0._r8) / (ci+kc*(1._r8+oair/ko))

         ! C3: RuBP-limited photosynthesis
         aj = je * max(ci-cp, 0._r8) / (4._r8*ci+8._r8*cp)

         ! C3: Product-limited photosynthesis
         ap = 3._r8 * tpu_z

      else

         ! C4: Rubisco-limited photosynthesis
         ac = vcmax_z

         ! C4: RuBP-limited photosynthesis
         aj = qe * par_z * 4.6_r8

         ! C4: PEP carboxylase-limited (CO2-limited)
         ap = kp_z * max(ci, 0._r8) / forc_pbot

      end if

      ! Gross photosynthesis. First co-limit ac and aj. Then co-limit ap

      aquad = theta_cj
      bquad = -(ac + aj)
      cquad = ac * aj
      call quadratic (aquad, bquad, cquad, r1, r2)
      ai = min(r1,r2)

      aquad = theta_ip
      bquad = -(ai + ap)
      cquad = ai * ap
      call quadratic (aquad, bquad, cquad, r1, r2)
      ag = max(0._r8,min(r1,r2))

      ! Net photosynthesis. Exit iteration if an < 0

      an = ag - lmr_z
      if (an < 0._r8) then
         fval = 0._r8
         return
      endif
      ! Quadratic gs_mol calculation with an known. Valid for an >= 0.
      ! With an <= 0, then gs_mol = bbb or medlyn intercept
      cs = cair - 1.4_r8/gb_mol * an * forc_pbot
      !cs = max(cs,max_cs)
      if ( stomatalcond_mtd == stomatalcond_mtd_medlyn2011 )then
          term = 1.6_r8 * an / (cs / forc_pbot * 1.e06_r8)
          aquad = 1.0_r8
          bquad = -(2.0 * (medlynintercept*1.e-06_r8 + term) + (medlynslope * term)**2 / &
               (gb_mol*1.e-06_r8 * rh_can))
          cquad = medlynintercept*medlynintercept*1.e-12_r8 + &
               (2.0*medlynintercept*1.e-06_r8 + term * &
               (1.0 - medlynslope* medlynslope / rh_can)) * term

          call quadratic (aquad, bquad, cquad, r1, r2)
          gs_mol = max(r1,r2) * 1.e06_r8
       else if ( stomatalcond_mtd == stomatalcond_mtd_bb1987 )then
          aquad = cs
          bquad = cs*(gb_mol - bbb) - mbb*an*forc_pbot
          cquad = -gb_mol*(cs*bbb + mbb*an*forc_pbot*rh_can)
          call quadratic (aquad, bquad, cquad, r1, r2)
          gs_mol = max(r1,r2)
       end if

      ! Derive new estimate for ci
      fval =ci - cair + an * forc_pbot * (1.4_r8*gs_mol+1.6_r8*gb_mol) / (gb_mol*gs_mol)
      
  end subroutine ci_func