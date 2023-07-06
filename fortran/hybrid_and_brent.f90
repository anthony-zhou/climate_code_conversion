subroutine hybrid(x0, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z,&
    rh_can, gs_mol,iter, &
    atm2lnd_inst, photosyns_inst)
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
type(atm2lnd_type)  , intent(in)    :: atm2lnd_inst
type(photosyns_type), intent(inout) :: photosyns_inst
!
!! LOCAL VARIABLES
real(r8) :: a, b
real(r8) :: fa, fb
real(r8) :: x1, f0, f1
real(r8) :: x, dx
real(r8), parameter :: eps = 1.e-2_r8      !relative accuracy
real(r8), parameter :: eps1= 1.e-4_r8
integer,  parameter :: itmax = 40          !maximum number of iterations
real(r8) :: tol,minx,minf

call ci_func(x0, f0, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z, rh_can, gs_mol, &
        atm2lnd_inst, photosyns_inst)

if(f0 == 0._r8)return

minx=x0
minf=f0
x1 = x0 * 0.99_r8

call ci_func(x1,f1, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z, rh_can, gs_mol, &
        atm2lnd_inst, photosyns_inst)

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

    call ci_func(x1,f1, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z, rh_can, gs_mol, &
        atm2lnd_inst, photosyns_inst)

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
            lmr_z, par_z, rh_can, gs_mol, &
            atm2lnd_inst, photosyns_inst)

        x0=x
        exit
    endif
    if(iter>itmax)then
        !in case of failing to converge within itmax iterations
        !stop at the minimum function
        !this happens because of some other issues besides the stomatal conductance calculation
        !and it happens usually in very dry places and more likely with c4 plants.

        call ci_func(minx,f1, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z, rh_can, gs_mol, &
            atm2lnd_inst, photosyns_inst)

        exit
    endif
enddo

end subroutine hybrid

!------------------------------------------------------------------------------
subroutine brent(x, x1,x2,f1, f2, tol, ip, iv, ic, gb_mol, je, cair, oair,&
    lmr_z, par_z, rh_can, gs_mol, &
    atm2lnd_inst, photosyns_inst)
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
type(atm2lnd_type)  , intent(in)    :: atm2lnd_inst
type(photosyns_type), intent(inout) :: photosyns_inst
!
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
    write(iulog,*) 'root must be bracketed for brent'
    call endrun(subgrid_index=ip, subgrid_level=subgrid_level_patch, msg=errmsg(sourcefile, 2398))
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

    call ci_func(b, fb, ip, iv, ic, gb_mol, je, cair, oair, lmr_z, par_z, rh_can, gs_mol, &
        atm2lnd_inst, photosyns_inst)

    if(fb==0._r8)exit

enddo

if(iter==itmax)write(iulog,*) 'brent exceeding maximum iterations', b, fb
x=b

return
end subroutine brent