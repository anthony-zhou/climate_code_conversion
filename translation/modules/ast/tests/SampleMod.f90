module SampleMod

  !------------------------------------------------------------------------------
  ! !DESCRIPTION:
  ! Sample module used for unit tests
  !
  implicit none
  ! !PRIVATE MEMBER FUNCTIONS:
  public :: ci_func        ! ci function
  integer,parameter :: r8 = selected_real_kind(12) ! 8 byte real

   contains

   real(r8) function add(a, b)
      real(r8), intent(in) :: a, b

      add = a + b
   end function add

   subroutine quadratic(a, b, c, r1, r2)
      implicit none
      real(r8), intent(in) :: a, b, c
      real(r8), intent(out) :: r1, r2
      
      ! !LOCAL VARIABLES:
      real(r8) :: discriminant
      real(r8) :: q

      discriminant = add(a, b)

      r1 = 0.0
      r2 = 1.0
  end subroutine quadratic

  !------------------------------------------------------------------------------
  subroutine ci_func(ci, fval, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z, rh_can, gs_mol)
    !
    !! DESCRIPTION:
    ! evaluate the function
    ! f(ci)=ci - (ca - (1.37rb+1.65rs))*patm*an
    !!ARGUMENTS:
    real(r8)             , intent(in)    :: ci       ! intracellular leaf CO2

    real(r8) :: r1, r2

    call quadratic(1.0, 1.0, 1.0, r1, r2)

    fval = 10.0
    
  end subroutine ci_func

 end module SampleMod
