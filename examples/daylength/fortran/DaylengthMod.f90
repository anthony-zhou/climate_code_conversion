module DaylengthMod

  !-----------------------------------------------------------------------
  ! !DESCRIPTION:
  ! Computes daylength
  !
! #include "shr_assert.h"  ## I'M THE ONE WHO COMMENTED THIS OUT
  integer,parameter :: r8 = selected_real_kind(12) ! 8 byte real

  type bounds_type
     integer :: begg, endg           ! beginning and ending gridcell index
     integer :: begl, endl           ! beginning and ending landunit index
     integer :: begc, endc           ! beginning and ending column index
     integer :: begp, endp           ! beginning and ending patch index
     integer :: begCohort, endCohort ! beginning and ending cohort indices
     integer :: level                ! whether defined on the proc or clump level
     integer :: clump_index          ! if defined on the clump level, this gives the clump index
  end type bounds_type
  type, public :: grc

     ! topological mapping functionality, local 1d gdc arrays
     real(r8), pointer :: area             (:) ! total land area, gridcell (km^2)
     real(r8), pointer :: lat              (:) ! latitude (radians)
     real(r8), pointer :: lon              (:) ! longitude (radians)
     real(r8), pointer :: latdeg           (:) ! latitude (degrees)
     real(r8), pointer :: londeg           (:) ! longitude (degrees)
     logical , pointer :: active           (:) ! just needed for symmetry with other subgrid types

     integer,  pointer :: nbedrock         (:) ! index of uppermost bedrock layer

     ! Daylength
     real(r8) , pointer :: max_dayl        (:) ! maximum daylength for this grid cell (s)
     real(r8) , pointer :: dayl            (:) ! daylength (seconds)
     real(r8) , pointer :: prev_dayl       (:) ! daylength from previous timestep (seconds)

     ! indices into landunit-level arrays for landunits in this grid cell (ispval implies
     ! this landunit doesn't exist on this grid cell) [1:max_lunit, begg:endg]
     ! (note that the spatial dimension is last here, in contrast to most 2-d variables;
     ! this is for efficiency, since most loops will go over g in the outer loop, and
     ! landunit type in the inner loop)
     integer , pointer :: landunit_indices (:,:) 

   contains

     procedure, public :: Init
     procedure, public :: Clean
     
  end type grc              
  !
  implicit none
  save
  private

  ! TODO(wjs, 2018-05-16) We should make this object-oriented, and move max_dayl, dayl
  ! and prev_dayl out of GridcellType into a new type defined here. Then can also move
  ! the hist_addfld1d calls for DAYL and PREV_DAYL to here.
  !
  ! !PUBLIC MEMBER FUNCTIONS:
  public :: InitDaylength     ! initialize daylength for all grid cells
  public :: UpdateDaylength   ! update daylength for all grid cells

  ! The following are public only to support unit testing, and shouldn't generally be
  ! called from outside this module.
  public :: daylength            ! function to compute daylength
  public :: ComputeMaxDaylength  ! compute max daylength for each grid cell
  !
  !-----------------------------------------------------------------------

  character(len=*), parameter, private :: sourcefile = &
       __FILE__

contains

  !-----------------------------------------------------------------------
  elemental real(r8) function daylength(lat, decl)
    !
    ! !DESCRIPTION:
    ! Computes daylength (in seconds)
    !
    ! Latitude and solar declination angle should both be specified in radians. decl must
    ! be strictly less than pi/2; lat must be less than pi/2 within a small tolerance.
    !
    ! !USES:
    use shr_infnan_mod, only : nan => shr_infnan_nan, &
                               assignment(=)
    use shr_const_mod , only : SHR_CONST_PI
    !
    ! !ARGUMENTS:
    real(r8), intent(in) :: lat    ! latitude (radians)
    real(r8), intent(in) :: decl   ! solar declination angle (radians)
    !
    ! !LOCAL VARIABLES:
    real(r8) :: my_lat             ! local version of lat, possibly adjusted slightly
    real(r8) :: temp               ! temporary variable

    ! number of seconds per radian of hour-angle
    real(r8), parameter :: secs_per_radian = 13750.9871_r8

    ! epsilon for defining latitudes "near" the pole
    real(r8), parameter :: lat_epsilon = 10._r8 * epsilon(1._r8)

    ! Define an offset pole as slightly less than pi/2 to avoid problems with cos(lat) being negative
    real(r8), parameter :: pole = SHR_CONST_PI/2.0_r8
    real(r8), parameter :: offset_pole = pole - lat_epsilon
    !-----------------------------------------------------------------------

    ! Can't SHR_ASSERT in an elemental function; instead, return a bad value if any
    ! preconditions are violated

    ! lat must be less than pi/2 within a small tolerance
    if (abs(lat) >= (pole + lat_epsilon)) then
       daylength = nan

    ! decl must be strictly less than pi/2
    else if (abs(decl) >= pole) then
       daylength = nan

    ! normal case
    else    
       ! Ensure that latitude isn't too close to pole, to avoid problems with cos(lat) being negative
       my_lat = min(offset_pole, max(-1._r8 * offset_pole, lat))

       temp = -(sin(my_lat)*sin(decl))/(cos(my_lat) * cos(decl))
       temp = min(1._r8,max(-1._r8,temp))
       daylength = 2.0_r8 * secs_per_radian * acos(temp) 
    end if

  end function daylength

  !-----------------------------------------------------------------------
  subroutine ComputeMaxDaylength(bounds, lat, obliquity, max_daylength)
    !
    ! !DESCRIPTION:
    ! Compute max daylength for each grid cell
    !
    ! !ARGUMENTS:
    type(bounds_type), intent(in) :: bounds
    real(r8), intent(in)  :: obliquity           ! earth's obliquity (radians)
    real(r8), intent(in)  :: lat(bounds%begg: )  ! latitude (radians)
    real(r8), intent(out) :: max_daylength(bounds%begg: ) ! maximum daylength for each gridcell (s)
    !
    ! !LOCAL VARIABLES:
    integer  :: g
    real(r8) :: max_decl  ! max declination angle

    character(len=*), parameter :: subname = 'ComputeMaxDaylength'
    !-----------------------------------------------------------------------

    SHR_ASSERT_ALL_FL((ubound(lat) == (/bounds%endg/)), sourcefile, __LINE__)
    SHR_ASSERT_ALL_FL((ubound(max_daylength) == (/bounds%endg/)), sourcefile, __LINE__)

    do g = bounds%begg,bounds%endg
       max_decl = obliquity
       if (lat(g) < 0._r8) max_decl = -max_decl
       max_daylength(g) = daylength(lat(g), max_decl)
    end do

  end subroutine ComputeMaxDaylength

  !-----------------------------------------------------------------------
  subroutine InitDaylength(bounds, declin, declinm1, obliquity)
    !
    ! !DESCRIPTION:
    ! Initialize daylength, previous daylength and max daylength for all grid cells.
    !
    ! This should be called with declin set at the value for the first model time step,
    ! and declinm1 at the value for the previous time step
    !
    ! !ARGUMENTS:
    type(bounds_type), intent(in) :: bounds
    real(r8), intent(in) :: declin              ! solar declination angle for the first model time step (radians)
    real(r8), intent(in) :: declinm1            ! solar declination angle for the previous time step (radians)
    real(r8), intent(in) :: obliquity           ! earth's obliquity (radians)
    !
    !-----------------------------------------------------------------------

    associate(&
    lat       => grc%lat,       & ! Input:   [real(r8) (:)] latitude (radians)
    dayl      => grc%dayl,      & ! Output:  [real(r8) (:)] day length (s)
    prev_dayl => grc%prev_dayl, & ! Output:  [real(r8) (:)] day length from previous time step (s)
    max_dayl  => grc%max_dayl , & ! Output:  [real(r8) (:)] maximum day length (s)

    begg      => bounds%begg  , & ! beginning grid cell index
    endg      => bounds%endg    & ! ending grid cell index
    )

    prev_dayl(begg:endg) = daylength(lat(begg:endg), declinm1)
    dayl(begg:endg) = daylength(lat(begg:endg), declin)

    call ComputeMaxDaylength(bounds, &
         lat = lat(bounds%begg:bounds%endg), &
         obliquity = obliquity, &
         max_daylength = max_dayl(bounds%begg:bounds%endg))

    end associate

  end subroutine InitDaylength


  !-----------------------------------------------------------------------
  subroutine UpdateDaylength(bounds, declin, obliquity)
    !
    ! !DESCRIPTION:
    ! Update daylength, previous daylength and max daylength for all grid cells.
    !
    ! This should be called exactly once per time step.
    !
    ! Assumes that InitDaylength has been called in initialization. This Update routine
    ! should NOT be called in initialization.
    !
    ! !USES:
    use clm_time_manager, only : is_first_step_of_this_run_segment
    !
    ! !ARGUMENTS:
    type(bounds_type), intent(in) :: bounds
    real(r8), intent(in) :: declin            ! solar declination angle (radians)
    real(r8), intent(in) :: obliquity         ! earth's obliquity (radians)
    !
    !-----------------------------------------------------------------------

    associate(&
    lat       => grc%lat,       & ! Input:  [real(r8) (:)] latitude (radians)
    dayl      => grc%dayl,      & ! InOut:  [real(r8) (:)] day length (s)
    prev_dayl => grc%prev_dayl, & ! Output: [real(r8) (:)] day length from previous time step (s)
    max_dayl  => grc%max_dayl , & ! Output: [real(r8) (:)] maximum day length (s)

    begg      => bounds%begg  , & ! beginning grid cell index
    endg      => bounds%endg    & ! ending grid cell index
    )

    if (is_first_step_of_this_run_segment()) then
       ! DO NOTHING
       !
       ! In the first time step, we simply use dayl & prev_dayl that were set in
       ! initialization. (We do NOT want to run the normal code in that case, because that
       ! would incorrectly set prev_dayl to be the same as the current dayl in the first
       ! time step, because of the way prev_dayl is initialized.)
    else 
       prev_dayl(begg:endg) = dayl(begg:endg)
       dayl(begg:endg) = daylength(lat(begg:endg), declin)
    end if

    call ComputeMaxDaylength(bounds, &
         lat = lat(bounds%begg:bounds%endg), &
         obliquity = obliquity, &
         max_daylength = max_dayl(bounds%begg:bounds%endg))

    end associate

  end subroutine UpdateDaylength

end module DaylengthMod