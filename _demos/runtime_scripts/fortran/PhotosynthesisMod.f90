module PhotosynthesisMod

  !------------------------------------------------------------------------------
  ! !DESCRIPTION:
  ! Leaf photosynthesis and stomatal conductance calculation as described by
  ! Bonan et al (2011) JGR, 116, doi:10.1029/2010JG001593 and extended to
  ! a multi-layer canopy
  !
  implicit none
  ! !PRIVATE MEMBER FUNCTIONS:
  public :: ci_func        ! ci function (CHANGED TO PUBLIC - alz)
  integer,parameter :: r8 = selected_real_kind(12) ! 8 byte real

   contains


   subroutine quadratic(a, b, c, r1, r2)
      implicit none
      real(r8), intent(in) :: a, b, c
      real(r8), intent(out) :: r1, r2
      
      ! !LOCAL VARIABLES:
      real(r8) :: discriminant
      real(r8) :: q

      discriminant = b * b - 4.0 * a * c


      if (a == 0._r8) then
         print *, "Quadratic solution error: a = 0."
         stop
      end if

      if (discriminant < 0.0) then
         if ( - discriminant < 3.0_r8*epsilon(b)) then
            discriminant = 0.0_r8
         else
            print *, "Quadratic solution error: b^2 - 4ac is negative."
            print *, a, b, c
            stop
         end if
      end if

      if (b >= 0._r8) then
         q = -0.5_r8 * (b + sqrt(discriminant))
      else
         q = -0.5_r8 * (b - sqrt(discriminant))
      end if
      
      r1 = q / a
      if (q /= 0._r8) then
         r2 = c / q
      else
         r2 = 1.e36_r8
      end if          
  end subroutine quadratic

  !------------------------------------------------------------------------------
  subroutine ci_func(ci, fval, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z, rh_can, gs_mol)
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
   medlynintercept = 10000._r8 ! Intercept for Medlyn stomatal conductance model method
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
       
      ! LRH: If the quadratic solver for gs_mol above doesn't work, try this:
      ! if ( stomatalcond_mtd == stomatalcond_mtd_medlyn2011 )then
      !        gs_mol = 1.6_r8 * (1._r8 + medlynslope / sqrt(2.0_r8)) * ( an / cs ) * 1.e06_r8! Medlyn
      ! else 
      !        gs_mol = medlynslope * rh_can * ( an / cs )   
      ! end if
      
      
      ! Derive new estimate for ci
      fval =ci - cair + an * forc_pbot * (1.4_r8*gs_mol+1.6_r8*gb_mol) / (gb_mol*gs_mol)
      
      ! Anthony: Save fval and gs_mol for evaluation

  end subroutine ci_func


  subroutine hybrid(x0, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z,&
   rh_can, gs_mol,iter)
!
!! DESCRIPTION:
! use a hybrid solver to find the root of equation
! f(x) = x- h(x),
!we want to find x, s.t. f(x) = 0.
!the hybrid approach combines the strength of the newton secant approach (find the solution domain)
!and the bisection approach implemented with the Brent's method to guarrantee convergence.

!
!! REVISION HISTORY:
!Dec 14/2012: created by Jinyun Tang
!
!!USES:
!
!! ARGUMENTS:
implicit none
real(r8), intent(inout) :: x0              !initial guess and final value of the solution
real(r8), intent(in) :: lmr_z              ! canopy layer: leaf maintenance respiration rate (umol CO2/m**2/s)
real(r8), intent(in) :: par_z              ! par absorbed per unit lai for canopy layer (w/m**2)
real(r8), intent(in) :: rh_can             ! canopy air relative humidity
real(r8), intent(in) :: gb_mol             ! leaf boundary layer conductance (umol H2O/m**2/s)
real(r8), intent(in) :: je                 ! electron transport rate (umol electrons/m**2/s)
real(r8), intent(in) :: cair               ! Atmospheric CO2 partial pressure (Pa)
real(r8), intent(in) :: oair               ! Atmospheric O2 partial pressure (Pa)
integer,  intent(in) :: p, iv, c           ! pft, c3/c4, and column index
real(r8), intent(out) :: gs_mol            ! leaf stomatal conductance (umol H2O/m**2/s)
integer,  intent(out) :: iter              !number of iterations used, for record only

!! LOCAL VARIABLES
real(r8) :: a, b
real(r8) :: fa, fb
real(r8) :: x1, f0, f1
real(r8) :: x, dx
real(r8), parameter :: eps = 1.e-2_r8      !relative accuracy
real(r8), parameter :: eps1= 1.e-4_r8
integer,  parameter :: itmax = 40          !maximum number of iterations
real(r8) :: tol,minx,minf

call ci_func(x0, f0, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z, rh_can, gs_mol)

if(f0 == 0._r8)return

minx=x0
minf=f0
x1 = x0 * 0.99_r8

call ci_func(x1,f1, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z, rh_can, gs_mol)

if(f1==0._r8)then
   x0 = x1
   return
endif
if(f1<minf)then
   minx=x1
   minf=f1
endif

!first use the secant approach, then use the brent approach as a backup
iter = 0
do
   iter = iter + 1
   dx = - f1 * (x1-x0)/(f1-f0)
   x = x1 + dx
   tol = abs(x) * eps
   if(abs(dx)<tol)then
       x0 = x
       exit
   endif
   x0 = x1
   f0 = f1
   x1 = x

   call ci_func(x1,f1, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z, rh_can, gs_mol)

   if(f1<minf)then
       minx=x1
       minf=f1
   endif
   if(abs(f1)<=eps1)then
       x0 = x1
       exit
   endif

   !if a root zone is found, use the brent method for a robust backup strategy
   if(f1 * f0 < 0._r8)then

       call brent(x, x0,x1,f0,f1, tol, p, iv, c, gb_mol, je, cair, oair, &
           lmr_z, par_z, rh_can, gs_mol)

       x0=x
       exit
   endif
   if(iter>itmax)then
       !in case of failing to converge within itmax iterations
       !stop at the minimum function
       !this happens because of some other issues besides the stomatal conductance calculation
       !and it happens usually in very dry places and more likely with c4 plants.

       call ci_func(minx,f1, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z, rh_can, gs_mol)

       exit
   endif
enddo

end subroutine hybrid

!------------------------------------------------------------------------------
subroutine brent(x, x1,x2,f1, f2, tol, ip, iv, ic, gb_mol, je, cair, oair,&
   lmr_z, par_z, rh_can, gs_mol)
!
!!DESCRIPTION:
!Use Brent's method to find the root of a single variable function ci_func, which is known to exist between x1 and x2.
!The found root will be updated until its accuracy is tol.

!!REVISION HISTORY:
!Dec 14/2012: Jinyun Tang, modified from numerical recipes in F90 by press et al. 1188-1189
!
!!ARGUMENTS:
real(r8), intent(out) :: x                ! indepedent variable of the single value function ci_func(x)
real(r8), intent(in) :: x1, x2, f1, f2    ! minimum and maximum of the variable domain to search for the solution ci_func(x1) = f1, ci_func(x2)=f2
real(r8), intent(in) :: tol               ! the error tolerance
real(r8), intent(in) :: lmr_z             ! canopy layer: leaf maintenance respiration rate (umol CO2/m**2/s)
real(r8), intent(in) :: par_z             ! par absorbed per unit lai for canopy layer (w/m**2)
real(r8), intent(in) :: gb_mol            ! leaf boundary layer conductance (umol H2O/m**2/s)
real(r8), intent(in) :: je                ! electron transport rate (umol electrons/m**2/s)
real(r8), intent(in) :: cair              ! Atmospheric CO2 partial pressure (Pa)
real(r8), intent(in) :: oair              ! Atmospheric O2 partial pressure (Pa)
real(r8), intent(in) :: rh_can            ! inside canopy relative humidity
integer,  intent(in) :: ip, iv, ic        ! pft, c3/c4, and column index
real(r8), intent(out) :: gs_mol           ! leaf stomatal conductance (umol H2O/m**2/s)

!!LOCAL VARIABLES:
integer, parameter :: itmax=20            !maximum number of iterations
real(r8), parameter :: eps=1.e-2_r8       !relative error tolerance
integer :: iter
real(r8)  :: a,b,c,d,e,fa,fb,fc,p,q,r,s,tol1,xm
!------------------------------------------------------------------------------

a=x1
b=x2
fa=f1
fb=f2
if((fa > 0._r8 .and. fb > 0._r8).or.(fa < 0._r8 .and. fb < 0._r8))then
   print *, "root must be bracketed for brent"
   stop
   ! write(iulog,*) 'root must be bracketed for brent'
   ! call endrun(subgrid_index=ip, subgrid_level=subgrid_level_patch, msg=errmsg(sourcefile, 2398))
endif
c=b
fc=fb
iter = 0
do
   if(iter==itmax)exit
   iter=iter+1
   if((fb > 0._r8 .and. fc > 0._r8) .or. (fb < 0._r8 .and. fc < 0._r8))then
       c=a   !Rename a, b, c and adjust bounding interval d.
       fc=fa
       d=b-a
       e=d
   endif
   if( abs(fc) < abs(fb)) then
       a=b
       b=c
       c=a
       fa=fb
       fb=fc
       fc=fa
   endif
   tol1=2._r8*eps*abs(b)+0.5_r8*tol  !Convergence check.
   xm=0.5_r8*(c-b)
   if(abs(xm) <= tol1 .or. fb == 0.)then
       x=b
       return
   endif
   if(abs(e) >= tol1 .and. abs(fa) > abs(fb)) then
       s=fb/fa !Attempt inverse quadratic interpolation.
       if(a == c) then
           p=2._r8*xm*s
           q=1._r8-s
       else
           q=fa/fc
           r=fb/fc
           p=s*(2._r8*xm*q*(q-r)-(b-a)*(r-1._r8))
           q=(q-1._r8)*(r-1._r8)*(s-1._r8)
       endif
       if(p > 0._r8) q=-q !Check whether in bounds.
       p=abs(p)
       if(2._r8*p < min(3._r8*xm*q-abs(tol1*q),abs(e*q))) then
           e=d !Accept interpolation.
           d=p/q
       else
           d=xm  !Interpolation failed, use bisection.
           e=d
       endif
   else !Bounds decreasing too slowly, use bisection.
       d=xm
       e=d
   endif
   a=b !Move last best guess to a.
   fa=fb
   if(abs(d) > tol1) then !Evaluate new trial root.
       b=b+d
   else
       b=b+sign(tol1,xm)
   endif

   call ci_func(b, fb, ip, iv, ic, gb_mol, je, cair, oair, lmr_z, par_z, rh_can, gs_mol)

   if(fb==0._r8)exit

enddo

if(iter==itmax)print *, "brent exceeding maximum iterations", b, fb
x=b

return
end subroutine brent


 end module PhotosynthesisMod
