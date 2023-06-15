

module  PhotosynthesisMod

use shr_assert_mod

    !------------------------------------------------------------------------------
    ! !DESCRIPTION:
    ! Leaf photosynthesis and stomatal conductance calculation as described by
    ! Bonan et al (2011) JGR, 116, doi:10.1029/2010JG001593 and extended to
    ! a multi-layer canopy
    !
    ! !USES:
    use shr_sys_mod         , only : shr_sys_flush
    use shr_kind_mod        , only : r8 => shr_kind_r8
    use shr_log_mod         , only : errMsg => shr_log_errMsg
    use shr_infnan_mod      , only : nan => shr_infnan_nan, assignment(=)
    use abortutils          , only : endrun
    use clm_varctl          , only : use_c13, use_c14, use_cn, use_cndv, use_fates, use_luna, use_hydrstress
    use clm_varctl          , only : iulog
    use clm_varpar          , only : nlevcan, nvegwcs, mxpft
    use clm_varcon          , only : c14ratio, spval, isecspday
    use decompMod           , only : bounds_type, subgrid_level_patch
    use QuadraticMod        , only : quadratic
    use pftconMod           , only : pftcon
    use CIsoAtmTimeseriesMod, only : C14BombSpike, use_c14_bombspike, C13TimeSeries, use_c13_timeseries, nsectors_c14
    use atm2lndType         , only : atm2lnd_type
    use CanopyStateType     , only : canopystate_type
    use WaterDiagnosticBulkType      , only : waterdiagnosticbulk_type
    use WaterFluxBulkType       , only : waterfluxbulk_type
    use SoilStateType       , only : soilstate_type
    use TemperatureType     , only : temperature_type
    use SolarAbsorbedType   , only : solarabs_type
    use SurfaceAlbedoType   , only : surfalb_type
    use OzoneBaseMod        , only : ozone_base_type
    use LandunitType        , only : lun
    use PatchType           , only : patch
    use GridcellType        , only : grc
    !
    implicit none
    private
    !
    ! !PUBLIC MEMBER FUNCTIONS:
    public :: Photosynthesis        ! Leaf stomatal resistance and leaf photosynthesis
    public :: PhotosynthesisTotal   ! Determine of total photosynthesis
    public :: Fractionation         ! C13 fractionation during photosynthesis
    ! For plant hydraulics approach
    public :: PhotosynthesisHydraulicStress ! Leaf stomatal resistance and leaf photosynthesis
                                            ! Simultaneous solution of sunlit/shaded per Pierre
                                            ! Gentine/Daniel Kennedy plant hydraulic stress method
    public :: plc                           ! Return value of vulnerability curve at x

    ! PRIVATE FUNCTIONS MADE PUBLIC Juse for unit-testing:
    public  :: d1plc          ! compute 1st deriv of conductance attenuation for each segment

    ! !PRIVATE MEMBER FUNCTIONS:
    private :: hybrid         ! hybrid solver for ci
    private :: ci_func        ! ci function
    private :: brent          ! brent solver for root of a single variable function
    private :: ft             ! photosynthesis temperature response
    private :: fth            ! photosynthesis temperature inhibition
    private :: fth25          ! scaling factor for photosynthesis temperature inhibition
    ! For plant hydraulics approach
    private :: hybrid_PHS     ! hybrid solver for ci
    private :: ci_func_PHS    ! ci function
    private :: brent_PHS      ! brent solver for root of a single variable function
    private :: calcstress     ! compute the root water stress
    private :: getvegwp       ! calculate vegetation water potential (sun, sha, xylem, root)
    private :: getqflx        ! calculate sunlit and shaded transpiration
    private :: spacF          ! flux divergence across each vegetation segment
    private :: spacA          ! the inverse Jacobian matrix relating delta(vegwp) to f, d(vegwp)=A*f

    ! !PRIVATE DATA:
    integer, parameter, private :: leafresp_mtd_ryan1991  = 1  ! Ryan 1991 method for lmr25top
    integer, parameter, private :: leafresp_mtd_atkin2015 = 2  ! Atkin 2015 method for lmr25top
    integer, parameter, private :: vegetation_weibull=0        ! PLC method type
    ! These are public for unit-tests
    integer, parameter, public   :: sun=1                 ! index for sunlit
    integer, parameter, public   :: sha=2                 ! index for shaded
    integer, parameter, public  :: xyl=3                  ! index for xylem
    integer, parameter, public  :: root=4                 ! index for root
    integer, parameter, public  :: veg=vegetation_weibull ! index for vegetation
    integer, parameter, public  :: soil=1                 ! index for soil
    integer, parameter, private :: stomatalcond_mtd_bb1987     = 1   ! Ball-Berry 1987 method for photosynthesis
    integer, parameter, private :: stomatalcond_mtd_medlyn2011 = 2   ! Medlyn 2011 method for photosynthesis

    real(r8), parameter, private :: bbbopt_c3 = 10000._r8            ! Ball-Berry Photosynthesis intercept to use for C3 vegetation
    real(r8), parameter, private :: bbbopt_c4 = 40000._r8            ! Ball-Berry Photosynthesis intercept to use for C4 vegetation
    real(r8), parameter, private :: medlyn_rh_can_max = 50._r8       ! Maximum to put on RH in the canopy used for Medlyn Photosynthesis
    real(r8), parameter, private :: medlyn_rh_can_fact = 0.001_r8    ! Multiplicitive factor to use for Canopy RH used for Medlyn photosynthesis
    real(r8), parameter, private :: max_cs = 1.e-06_r8               ! Max CO2 partial pressure at leaf surface (Pa) for PHS
    ! !PUBLIC VARIABLES:

    type :: photo_params_type
        real(r8) :: act25  ! Rubisco activity at 25 C (umol CO2/gRubisco/s)
        real(r8) :: fnr  ! Mass ratio of total Rubisco molecular mass to nitrogen in Rubisco (gRubisco/gN in Rubisco)
        real(r8) :: cp25_yr2000  ! CO2 compensation point at 25°C at present day O2 (mol/mol)
        real(r8) :: kc25_coef  ! Michaelis-Menten const. at 25°C for CO2 (unitless)
        real(r8) :: ko25_coef  ! Michaelis-Menten const. at 25°C for O2 (unitless)
        real(r8) :: fnps       ! Fraction of light absorbed by non-photosynthetic pigment (unitless)
        real(r8) :: theta_psii ! Empirical curvature parameter for electron transport rate (unitless)
        real(r8) :: theta_ip   ! Empirical curvature parameter for ap photosynthesis co-limitation (unitless)
        real(r8) :: vcmaxha    ! Activation energy for vcmax (J/mol)
        real(r8) :: jmaxha     ! Activation energy for jmax (J/mol)
        real(r8) :: tpuha      ! Activation energy for tpu (J/mol)
        real(r8) :: lmrha      ! Activation energy for lmr (J/mol)
        real(r8) :: kcha       ! Activation energy for kc (J/mol)
        real(r8) :: koha       ! Activation energy for ko (J/mol)
        real(r8) :: cpha       ! Activation energy for cp (J/mol)
        real(r8) :: vcmaxhd    ! Deactivation energy for vcmax (J/mol)
        real(r8) :: jmaxhd     ! Deactivation energy for jmax (J/mol)
        real(r8) :: tpuhd      ! Deactivation energy for tpu (J/mol)
        real(r8) :: lmrhd      ! Deactivation energy for lmr (J/mol)
        real(r8) :: lmrse      ! Entropy term for lmr (J/mol/K)
        real(r8) :: tpu25ratio ! Ratio of tpu25top to vcmax25top (unitless)
        real(r8) :: kp25ratio  ! Ratio of kp25top to vcmax25top (unitless)
        real(r8) :: vcmaxse_sf ! Scale factor for vcmaxse (unitless)
        real(r8) :: jmaxse_sf  ! Scale factor for jmaxse (unitless)
        real(r8) :: tpuse_sf   ! Scale factor for tpuse (unitless)
        real(r8) :: jmax25top_sf ! Scale factor for jmax25top (unitless)
        real(r8), allocatable, public  :: krmax              (:)
        real(r8), allocatable, private :: kmax               (:,:)
        real(r8), allocatable, private :: psi50              (:,:)
        real(r8), allocatable, private :: ck                 (:,:)
        real(r8), allocatable, private :: lmr_intercept_atkin(:)
        real(r8), allocatable, private :: theta_cj           (:) ! Empirical curvature parameter for ac, aj photosynthesis co-limitation (unitless)
    contains
        procedure, private :: allocParams    ! Allocate the parameters
        procedure, private :: cleanParams    ! Deallocate parameters from member
    end type photo_params_type
    !
    type(photo_params_type), public, protected :: params_inst  ! params_inst is populated in readParamsMod 

    type, public :: photosyns_type

        logical , pointer, private :: c3flag_patch      (:)   ! patch true if C3 and false if C4
        ! Plant hydraulic stress specific variables
        real(r8), pointer, private :: ac_phs_patch      (:,:,:) ! patch Rubisco-limited gross photosynthesis (umol CO2/m**2/s)
        real(r8), pointer, private :: aj_phs_patch      (:,:,:) ! patch RuBP-limited gross photosynthesis (umol CO2/m**2/s)
        real(r8), pointer, private :: ap_phs_patch      (:,:,:) ! patch product-limited (C3) or CO2-limited (C4) gross photosynthesis (umol CO2/m**2/s)
        real(r8), pointer, private :: ag_phs_patch      (:,:,:) ! patch co-limited gross leaf photosynthesis (umol CO2/m**2/s)
        real(r8), pointer, private :: an_sun_patch      (:,:)   ! patch sunlit net leaf photosynthesis (umol CO2/m**2/s)
        real(r8), pointer, private :: an_sha_patch      (:,:)   ! patch shaded net leaf photosynthesis (umol CO2/m**2/s)
        real(r8), pointer, private :: vcmax_z_phs_patch (:,:,:) ! patch maximum rate of carboxylation (umol co2/m**2/s)
        real(r8), pointer, private :: kp_z_phs_patch    (:,:,:) ! patch initial slope of CO2 response curve (C4 plants)
        real(r8), pointer, private :: tpu_z_phs_patch   (:,:,:) ! patch triose phosphate utilization rate (umol CO2/m**2/s)
        real(r8), pointer, public  :: gs_mol_sun_patch  (:,:) ! patch sunlit leaf stomatal conductance (umol H2O/m**2/s)
        real(r8), pointer, public  :: gs_mol_sha_patch  (:,:) ! patch shaded leaf stomatal conductance (umol H2O/m**2/s)
        real(r8), pointer, private :: gs_mol_sun_ln_patch (:,:) ! patch sunlit leaf stomatal conductance averaged over 1 hour before to 1 hour after local noon (umol H2O/m**2/s)
        real(r8), pointer, private :: gs_mol_sha_ln_patch (:,:) ! patch shaded leaf stomatal conductance averaged over 1 hour before to 1 hour after local noon (umol H2O/m**2/s)
        real(r8), pointer, private :: ac_patch          (:,:) ! patch Rubisco-limited gross photosynthesis (umol CO2/m**2/s)
        real(r8), pointer, private :: aj_patch          (:,:) ! patch RuBP-limited gross photosynthesis (umol CO2/m**2/s)
        real(r8), pointer, private :: ap_patch          (:,:) ! patch product-limited (C3) or CO2-limited (C4) gross photosynthesis (umol CO2/m**2/s)
        real(r8), pointer, private :: ag_patch          (:,:) ! patch co-limited gross leaf photosynthesis (umol CO2/m**2/s)
        real(r8), pointer, private :: an_patch          (:,:) ! patch net leaf photosynthesis (umol CO2/m**2/s)
        real(r8), pointer, private :: vcmax_z_patch     (:,:) ! patch maximum rate of carboxylation (umol co2/m**2/s)
        real(r8), pointer, private :: cp_patch          (:)   ! patch CO2 compensation point (Pa)
        real(r8), pointer, private :: kc_patch          (:)   ! patch Michaelis-Menten constant for CO2 (Pa)
        real(r8), pointer, private :: ko_patch          (:)   ! patch Michaelis-Menten constant for O2 (Pa)
        real(r8), pointer, private :: qe_patch          (:)   ! patch quantum efficiency, used only for C4 (mol CO2 / mol photons)
        real(r8), pointer, private :: tpu_z_patch       (:,:) ! patch triose phosphate utilization rate (umol CO2/m**2/s)
        real(r8), pointer, private :: kp_z_patch        (:,:) ! patch initial slope of CO2 response curve (C4 plants)
        real(r8), pointer, private :: bbb_patch         (:)   ! patch Ball-Berry minimum leaf conductance (umol H2O/m**2/s)
        real(r8), pointer, private :: mbb_patch         (:)   ! patch Ball-Berry slope of conductance-photosynthesis relationship
        real(r8), pointer, private :: gs_mol_patch      (:,:) ! patch leaf stomatal conductance       (umol H2O/m**2/s)
        real(r8), pointer, private :: gb_mol_patch      (:)   ! patch leaf boundary layer conductance (umol H2O/m**2/s)
        real(r8), pointer, private :: rh_leaf_patch     (:)   ! patch fractional humidity at leaf surface (dimensionless)
        real(r8), pointer, private :: vpd_can_patch     (:)   ! patch canopy vapor pressure deficit (kPa)
        real(r8), pointer, private :: alphapsnsun_patch (:)   ! patch sunlit 13c fractionation ([])
        real(r8), pointer, private :: alphapsnsha_patch (:)   ! patch shaded 13c fractionation ([])

        real(r8), pointer, public  :: rc13_canair_patch (:)   ! patch C13O2/C12O2 in canopy air
        real(r8), pointer, public  :: rc13_psnsun_patch (:)   ! patch C13O2/C12O2 in sunlit canopy psn flux
        real(r8), pointer, public  :: rc13_psnsha_patch (:)   ! patch C13O2/C12O2 in shaded canopy psn flux

        real(r8), pointer, public  :: psnsun_patch      (:)   ! patch sunlit leaf photosynthesis     (umol CO2/m**2/s)
        real(r8), pointer, public  :: psnsha_patch      (:)   ! patch shaded leaf photosynthesis     (umol CO2/m**2/s)
        real(r8), pointer, public  :: c13_psnsun_patch  (:)   ! patch c13 sunlit leaf photosynthesis (umol 13CO2/m**2/s)
        real(r8), pointer, public  :: c13_psnsha_patch  (:)   ! patch c13 shaded leaf photosynthesis (umol 13CO2/m**2/s)
        real(r8), pointer, public  :: c14_psnsun_patch  (:)   ! patch c14 sunlit leaf photosynthesis (umol 14CO2/m**2/s)
        real(r8), pointer, public  :: c14_psnsha_patch  (:)   ! patch c14 shaded leaf photosynthesis (umol 14CO2/m**2/s)

        real(r8), pointer, private :: psnsun_z_patch    (:,:) ! patch canopy layer: sunlit leaf photosynthesis   (umol CO2/m**2/s)
        real(r8), pointer, private :: psnsha_z_patch    (:,:) ! patch canopy layer: shaded leaf photosynthesis   (umol CO2/m**2/s)
        real(r8), pointer, private :: psnsun_wc_patch   (:)   ! patch Rubsico-limited sunlit leaf photosynthesis (umol CO2/m**2/s)
        real(r8), pointer, private :: psnsha_wc_patch   (:)   ! patch Rubsico-limited shaded leaf photosynthesis (umol CO2/m**2/s)
        real(r8), pointer, private :: psnsun_wj_patch   (:)   ! patch RuBP-limited sunlit leaf photosynthesis    (umol CO2/m**2/s)
        real(r8), pointer, private :: psnsha_wj_patch   (:)   ! patch RuBP-limited shaded leaf photosynthesis    (umol CO2/m**2/s)
        real(r8), pointer, private :: psnsun_wp_patch   (:)   ! patch product-limited sunlit leaf photosynthesis (umol CO2/m**2/s)
        real(r8), pointer, private :: psnsha_wp_patch   (:)   ! patch product-limited shaded leaf photosynthesis (umol CO2/m**2/s)

        real(r8), pointer, public  :: fpsn_patch        (:)   ! patch photosynthesis                 (umol CO2/m**2 ground/s)
        real(r8), pointer, private :: fpsn_wc_patch     (:)   ! patch Rubisco-limited photosynthesis (umol CO2/m**2 ground/s)
        real(r8), pointer, private :: fpsn_wj_patch     (:)   ! patch RuBP-limited photosynthesis    (umol CO2/m**2 ground/s)
        real(r8), pointer, private :: fpsn_wp_patch     (:)   ! patch product-limited photosynthesis (umol CO2/m**2 ground/s)

        real(r8), pointer, public  :: lnca_patch        (:)   ! top leaf layer leaf N concentration (gN leaf/m^2)

        real(r8), pointer, public  :: lmrsun_patch      (:)   ! patch sunlit leaf maintenance respiration rate               (umol CO2/m**2/s)
        real(r8), pointer, public  :: lmrsha_patch      (:)   ! patch shaded leaf maintenance respiration rate               (umol CO2/m**2/s)
        real(r8), pointer, private :: lmrsun_z_patch    (:,:) ! patch canopy layer: sunlit leaf maintenance respiration rate (umol CO2/m**2/s)
        real(r8), pointer, private :: lmrsha_z_patch    (:,:) ! patch canopy layer: shaded leaf maintenance respiration rate (umol CO2/m**2/s)

        real(r8), pointer, public  :: cisun_z_patch     (:,:) ! patch intracellular sunlit leaf CO2 (Pa)
        real(r8), pointer, public  :: cisha_z_patch     (:,:) ! patch intracellular shaded leaf CO2 (Pa)

        real(r8), pointer, private :: rssun_z_patch     (:,:) ! patch canopy layer: sunlit leaf stomatal resistance (s/m)
        real(r8), pointer, private :: rssha_z_patch     (:,:) ! patch canopy layer: shaded leaf stomatal resistance (s/m)
        real(r8), pointer, public  :: rssun_patch       (:)   ! patch sunlit stomatal resistance (s/m)
        real(r8), pointer, public  :: rssha_patch       (:)   ! patch shaded stomatal resistance (s/m)
        real(r8), pointer, public  :: luvcmax25top_patch (:)   ! vcmax25 !     (umol/m2/s)
        real(r8), pointer, public  :: lujmax25top_patch  (:)   ! vcmax25 (umol/m2/s)
        real(r8), pointer, public  :: lutpu25top_patch   (:)   ! vcmax25 (umol/m2/s)
!!


        ! LUNA specific variables
        real(r8), pointer, public  :: vcmx25_z_patch    (:,:) ! patch  leaf Vc,max25 (umol CO2/m**2/s) for canopy layer 
        real(r8), pointer, public  :: jmx25_z_patch     (:,:) ! patch  leaf Jmax25 (umol electron/m**2/s) for canopy layer 
        real(r8), pointer, public  :: vcmx25_z_last_valid_patch  (:,:) ! patch  leaf Vc,max25 at the end of the growing season for the previous year
        real(r8), pointer, public  :: jmx25_z_last_valid_patch   (:,:) ! patch  leaf Jmax25 at the end of the growing season for the previous year
        real(r8), pointer, public  :: pnlc_z_patch      (:,:) ! patch proportion of leaf nitrogen allocated for light capture for canopy layer
        real(r8), pointer, public  :: enzs_z_patch      (:,:) ! enzyme decay status 1.0-fully active; 0-all decayed during stress
        real(r8), pointer, public  :: fpsn24_patch      (:)   ! 24 hour mean patch photosynthesis (umol CO2/m**2 ground/day)

        ! Logical switches for different options
        logical, public  :: rootstem_acc                      ! Respiratory acclimation for roots and stems
        logical, private :: light_inhibit                     ! If light should inhibit respiration
        integer, private :: leafresp_method                   ! leaf maintencence respiration at 25C for canopy top method to use
        integer, private :: stomatalcond_mtd                  ! Stomatal conduction method type
        logical, private :: modifyphoto_and_lmr_forcrop       ! Modify photosynthesis and LMR for crop
    contains

        ! Public procedures
        procedure, public  :: Init
        procedure, public  :: Restart
        procedure, public  :: ReadNML
        procedure, public  :: ReadParams
        procedure, public  :: TimeStepInit
        procedure, public  :: NewPatchInit
        procedure, public  :: Clean

        ! Procedures for unit-testing
        procedure, public  :: SetParamsForTesting

        ! Private procedures
        procedure, private :: InitAllocate
        procedure, private :: InitHistory
        procedure, private :: InitCold

    end type photosyns_type

    character(len=*), parameter, private :: sourcefile = &
        "./examples/photosynthesis/PhotosynthesisMod.f90"
    !------------------------------------------------------------------------

contains

    

    !------------------------------------------------------------------------------
    !------------------------------------------------------------------------------
    subroutine Photosynthesis ( bounds, fn, filterp, &
        esat_tv, eair, oair, cair, rb, btran, &
        dayl_factor, leafn, &
        atm2lnd_inst, temperature_inst, surfalb_inst, solarabs_inst, &
        canopystate_inst, ozone_inst, photosyns_inst, phase)
    !
    ! !DESCRIPTION:
    ! Leaf photosynthesis and stomatal conductance calculation as described by
    ! Bonan et al (2011) JGR, 116, doi:10.1029/2010JG001593 and extended to
    ! a multi-layer canopy
    !
    ! !USES:
    use clm_varcon        , only : rgas, tfrz, spval
    use GridcellType      , only : grc
    use clm_time_manager  , only : get_step_size_real, is_near_local_noon
    use clm_varctl     , only : cnallocate_carbon_only
    use clm_varctl     , only : lnc_opt, reduce_dayl_factor, vcmax_opt    
    use pftconMod      , only : nbrdlf_dcd_tmp_shrub, npcropmin

    !
    ! !ARGUMENTS:
    type(bounds_type)      , intent(in)    :: bounds
    integer                , intent(in)    :: fn                             ! size of pft filter
    integer                , intent(in)    :: filterp(fn)                    ! patch filter
    real(r8)               , intent(in)    :: esat_tv( bounds%begp: )        ! saturation vapor pressure at t_veg (Pa) [pft]
    real(r8)               , intent(in)    :: eair( bounds%begp: )           ! vapor pressure of canopy air (Pa) [pft]
    real(r8)               , intent(in)    :: oair( bounds%begp: )           ! Atmospheric O2 partial pressure (Pa) [pft]
    real(r8)               , intent(in)    :: cair( bounds%begp: )           ! Atmospheric CO2 partial pressure (Pa) [pft]
    real(r8)               , intent(in)    :: rb( bounds%begp: )             ! boundary layer resistance (s/m) [pft]
    real(r8)               , intent(in)    :: btran( bounds%begp: )          ! transpiration wetness factor (0 to 1) [pft]
    real(r8)               , intent(in)    :: dayl_factor( bounds%begp: )    ! scalar (0-1) for daylength
    real(r8)               , intent(in)    :: leafn( bounds%begp: )          ! leaf N (gN/m2)
    type(atm2lnd_type)     , intent(in)    :: atm2lnd_inst
    type(temperature_type) , intent(in)    :: temperature_inst
    type(surfalb_type)     , intent(in)    :: surfalb_inst
    type(solarabs_type)    , intent(in)    :: solarabs_inst
    type(canopystate_type) , intent(in)    :: canopystate_inst
    class(ozone_base_type) , intent(in)    :: ozone_inst
    type(photosyns_type)   , intent(inout) :: photosyns_inst
    character(len=*)       , intent(in)    :: phase                          ! 'sun' or 'sha'

    !
    ! !LOCAL VARIABLES:
    !
    ! Leaf photosynthesis parameters
    real(r8) :: jmax_z(bounds%begp:bounds%endp,nlevcan)  ! maximum electron transport rate (umol electrons/m**2/s)
    !real(r8) :: lnc(bounds%begp:bounds%endp)   ! leaf N concentration (gN leaf/m^2)
    real(r8) :: bbbopt(bounds%begp:bounds%endp)! Ball-Berry minimum leaf conductance, unstressed (umol H2O/m**2/s)
    real(r8) :: kn(bounds%begp:bounds%endp)    ! leaf nitrogen decay coefficient
    real(r8) :: vcmax25top     ! canopy top: maximum rate of carboxylation at 25C (umol CO2/m**2/s)
    real(r8) :: jmax25top      ! canopy top: maximum electron transport rate at 25C (umol electrons/m**2/s)
    real(r8) :: tpu25top       ! canopy top: triose phosphate utilization rate at 25C (umol CO2/m**2/s)
    real(r8) :: lmr25top       ! canopy top: leaf maintenance respiration rate at 25C (umol CO2/m**2/s)
    real(r8) :: kp25top        ! canopy top: initial slope of CO2 response curve (C4 plants) at 25C

    real(r8) :: vcmax25        ! leaf layer: maximum rate of carboxylation at 25C (umol CO2/m**2/s)
    real(r8) :: jmax25         ! leaf layer: maximum electron transport rate at 25C (umol electrons/m**2/s)
    real(r8) :: tpu25          ! leaf layer: triose phosphate utilization rate at 25C (umol CO2/m**2/s)
    real(r8) :: lmr25          ! leaf layer: leaf maintenance respiration rate at 25C (umol CO2/m**2/s)
    real(r8) :: kp25           ! leaf layer: Initial slope of CO2 response curve (C4 plants) at 25C
    real(r8) :: kc25           ! Michaelis-Menten constant for CO2 at 25C (Pa)
    real(r8) :: ko25           ! Michaelis-Menten constant for O2 at 25C (Pa)
    real(r8) :: cp25           ! CO2 compensation point at 25C (Pa)

    real(r8) :: vcmaxse        ! entropy term for vcmax (J/mol/K)
    real(r8) :: jmaxse         ! entropy term for jmax (J/mol/K)
    real(r8) :: tpuse          ! entropy term for tpu (J/mol/K)

    real(r8) :: vcmaxc         ! scaling factor for high temperature inhibition (25 C = 1.0)
    real(r8) :: jmaxc          ! scaling factor for high temperature inhibition (25 C = 1.0)
    real(r8) :: tpuc           ! scaling factor for high temperature inhibition (25 C = 1.0)
    real(r8) :: lmrc           ! scaling factor for high temperature inhibition (25 C = 1.0)

    ! Other
    integer  :: f,p,c,iv          ! indices
    real(r8) :: cf                ! s m**2/umol -> s/m
    real(r8) :: rsmax0            ! maximum stomatal resistance [s/m]
    real(r8) :: gb                ! leaf boundary layer conductance (m/s)
    real(r8) :: cs                ! CO2 partial pressure at leaf surface (Pa)
    real(r8) :: gs                ! leaf stomatal conductance (m/s)
    real(r8) :: hs                ! fractional humidity at leaf surface (dimensionless)
    real(r8) :: sco               ! relative specificity of rubisco
    real(r8) :: ft                ! photosynthesis temperature response (statement function)
    real(r8) :: fth               ! photosynthesis temperature inhibition (statement function)
    real(r8) :: fth25             ! ccaling factor for photosynthesis temperature inhibition (statement function)
    real(r8) :: tl                ! leaf temperature in photosynthesis temperature function (K)
    real(r8) :: ha                ! activation energy in photosynthesis temperature function (J/mol)
    real(r8) :: hd                ! deactivation energy in photosynthesis temperature function (J/mol)
    real(r8) :: se                ! entropy term in photosynthesis temperature function (J/mol/K)
    real(r8) :: scaleFactor       ! scaling factor for high temperature inhibition (25 C = 1.0)
    real(r8) :: ciold             ! previous value of Ci for convergence check
    real(r8) :: gs_mol_err        ! gs_mol for error check
    real(r8) :: je                ! electron transport rate (umol electrons/m**2/s)
    real(r8) :: qabs              ! PAR absorbed by PS II (umol photons/m**2/s)
    real(r8) :: aquad,bquad,cquad ! terms for quadratic equations
    real(r8) :: r1,r2             ! roots of quadratic equation
    real(r8) :: ceair             ! vapor pressure of air, constrained (Pa)
    integer  :: niter             ! iteration loop index
    real(r8) :: nscaler           ! leaf nitrogen scaling coefficient

    real(r8) :: ai                ! intermediate co-limited photosynthesis (umol CO2/m**2/s)

    real(r8) :: psn_wc_z(bounds%begp:bounds%endp,nlevcan) ! Rubisco-limited contribution to psn_z (umol CO2/m**2/s)
    real(r8) :: psn_wj_z(bounds%begp:bounds%endp,nlevcan) ! RuBP-limited contribution to psn_z (umol CO2/m**2/s)
    real(r8) :: psn_wp_z(bounds%begp:bounds%endp,nlevcan) ! product-limited contribution to psn_z (umol CO2/m**2/s)

    real(r8) :: psncan            ! canopy sum of psn_z
    real(r8) :: psncan_wc         ! canopy sum of psn_wc_z
    real(r8) :: psncan_wj         ! canopy sum of psn_wj_z
    real(r8) :: psncan_wp         ! canopy sum of psn_wp_z
    real(r8) :: lmrcan            ! canopy sum of lmr_z
    real(r8) :: gscan             ! canopy sum of leaf conductance
    real(r8) :: laican            ! canopy sum of lai_z
    real(r8) :: rh_can
    real(r8) , pointer :: lai_z       (:,:)
    real(r8) , pointer :: par_z       (:,:)
    real(r8) , pointer :: vcmaxcint   (:)
    real(r8) , pointer :: alphapsn    (:)
    real(r8) , pointer :: psn         (:)
    real(r8) , pointer :: psn_wc      (:)
    real(r8) , pointer :: psn_wj      (:)
    real(r8) , pointer :: psn_wp      (:)
    real(r8) , pointer :: psn_z       (:,:)
    real(r8) , pointer :: lmr         (:)
    real(r8) , pointer :: lmr_z       (:,:)
    real(r8) , pointer :: rs          (:)
    real(r8) , pointer :: rs_z        (:,:)
    real(r8) , pointer :: ci_z        (:,:)
    real(r8) , pointer :: o3coefv     (:)  ! o3 coefficient used in photo calculation
    real(r8) , pointer :: o3coefg     (:)  ! o3 coefficient used in rs calculation
    real(r8) , pointer :: alphapsnsun (:)
    real(r8) , pointer :: alphapsnsha (:)

    real(r8) :: sum_nscaler              
    real(r8) :: total_lai                
    integer  :: nptreemax                

    real(r8) :: dtime                           ! land model time step (sec)
    integer  :: g                               ! index
    !------------------------------------------------------------------------------

    ! Temperature and soil water response functions

    ft(tl,ha) = exp( ha / (rgas*1.e-3_r8*(tfrz+25._r8)) * (1._r8 - (tfrz+25._r8)/tl) )
    fth(tl,hd,se,scaleFactor) = scaleFactor / ( 1._r8 + exp( (-hd+se*tl) / (rgas*1.e-3_r8*tl) ) )
    fth25(hd,se) = 1._r8 + exp( (-hd+se*(tfrz+25._r8)) / (rgas*1.e-3_r8*(tfrz+25._r8)) )

    ! Enforce expected array sizes

    call shr_assert_all((ubound(esat_tv)     == (/bounds%endp/)), file= sourcefile, line= 1369)
    call shr_assert_all((ubound(eair)        == (/bounds%endp/)), file= sourcefile, line= 1370)
    call shr_assert_all((ubound(oair)        == (/bounds%endp/)), file= sourcefile, line= 1371)
    call shr_assert_all((ubound(cair)        == (/bounds%endp/)), file= sourcefile, line= 1372)
    call shr_assert_all((ubound(rb)          == (/bounds%endp/)), file= sourcefile, line= 1373)
    call shr_assert_all((ubound(btran)       == (/bounds%endp/)), file= sourcefile, line= 1374)
    call shr_assert_all((ubound(dayl_factor) == (/bounds%endp/)), file= sourcefile, line= 1375)
    call shr_assert_all((ubound(leafn)       == (/bounds%endp/)), file= sourcefile, line= 1376)

    associate(                                                 &
            c3psn      => pftcon%c3psn                          , & ! Input:  photosynthetic pathway: 0. = c4, 1. = c3
        crop       => pftcon%crop                           , & ! Input:  crop or not (0 =not crop and 1 = crop)
            leafcn     => pftcon%leafcn                         , & ! Input:  leaf C:N (gC/gN)
            flnr       => pftcon%flnr                           , & ! Input:  fraction of leaf N in the Rubisco enzyme (gN Rubisco / gN leaf)
            fnitr      => pftcon%fnitr                          , & ! Input:  foliage nitrogen limitation factor (-)
            slatop     => pftcon%slatop                         , & ! Input:  specific leaf area at top of canopy, projected area basis [m^2/gC]
            dsladlai   => pftcon%dsladlai                       , & ! Input:  change in sla per unit lai  
            i_vcad     => pftcon%i_vcad                         , & ! Input:  [real(r8) (:)   ]  
            s_vcad     => pftcon%s_vcad                         , & ! Input:  [real(r8) (:)   ]  
            i_flnr     => pftcon%i_flnr                         , & ! Input:  [real(r8) (:)   ]  
            s_flnr     => pftcon%s_flnr                         , & ! Input:  [real(r8) (:)   ]  
            mbbopt     => pftcon%mbbopt                         , & ! Input:  [real(r8) (:)   ]  Ball-Berry slope of conduct/photosyn (umol H2O/umol CO2)
            ivt        => patch%itype                           , & ! Input:  [integer  (:)   ]  patch vegetation type
            forc_pbot  => atm2lnd_inst%forc_pbot_downscaled_col , & ! Input:  [real(r8) (:)   ]  atmospheric pressure (Pa)

            t_veg      => temperature_inst%t_veg_patch          , & ! Input:  [real(r8) (:)   ]  vegetation temperature (Kelvin)
            t10        => temperature_inst%t_a10_patch          , & ! Input:  [real(r8) (:)   ]  10-day running mean of the 2 m temperature (K)
            tgcm       => temperature_inst%thm_patch            , & ! Input:  [real(r8) (:)   ]  air temperature at agcm reference height (kelvin)

            nrad       => surfalb_inst%nrad_patch               , & ! Input:  [integer  (:)   ]  pft number of canopy layers, above snow for radiative transfer
            tlai_z     => surfalb_inst%tlai_z_patch             , & ! Input:  [real(r8) (:,:) ]  pft total leaf area index for canopy layer
            tlai       => canopystate_inst%tlai_patch           , & ! Input:  [real(r8)(:)    ]  one-sided leaf area index, no burying by snow  
            c3flag     => photosyns_inst%c3flag_patch           , & ! Output: [logical  (:)   ]  true if C3 and false if C4
            ac         => photosyns_inst%ac_patch               , & ! Output: [real(r8) (:,:) ]  Rubisco-limited gross photosynthesis (umol CO2/m**2/s)
            aj         => photosyns_inst%aj_patch               , & ! Output: [real(r8) (:,:) ]  RuBP-limited gross photosynthesis (umol CO2/m**2/s)
            ap         => photosyns_inst%ap_patch               , & ! Output: [real(r8) (:,:) ]  product-limited (C3) or CO2-limited (C4) gross photosynthesis (umol CO2/m**2/s)
            ag         => photosyns_inst%ag_patch               , & ! Output: [real(r8) (:,:) ]  co-limited gross leaf photosynthesis (umol CO2/m**2/s)
            an         => photosyns_inst%an_patch               , & ! Output: [real(r8) (:,:) ]  net leaf photosynthesis (umol CO2/m**2/s)
            gb_mol     => photosyns_inst%gb_mol_patch           , & ! Output: [real(r8) (:)   ]  leaf boundary layer conductance (umol H2O/m**2/s)
            gs_mol     => photosyns_inst%gs_mol_patch           , & ! Output: [real(r8) (:,:) ]  leaf stomatal conductance (umol H2O/m**2/s)
            gs_mol_sun_ln => photosyns_inst%gs_mol_sun_ln_patch , & ! Output: [real(r8) (:,:) ]  sunlit leaf stomatal conductance averaged over 1 hour before to 1 hour after local noon (umol H2O/m**2/s)
            gs_mol_sha_ln => photosyns_inst%gs_mol_sha_ln_patch , & ! Output: [real(r8) (:,:) ]  shaded leaf stomatal conductance averaged over 1 hour before to 1 hour after local noon (umol H2O/m**2/s)
            gs_mol_sun => photosyns_inst%gs_mol_sun_patch       , & ! Output: [real(r8) (:,:) ]  patch sunlit leaf stomatal conductance (umol H2O/m**2/s)
            gs_mol_sha => photosyns_inst%gs_mol_sha_patch       , & ! Output: [real(r8) (:,:) ]  patch shaded leaf stomatal conductance (umol H2O/m**2/s)
            vcmax_z    => photosyns_inst%vcmax_z_patch          , & ! Output: [real(r8) (:,:) ]  maximum rate of carboxylation (umol co2/m**2/s)
            cp         => photosyns_inst%cp_patch               , & ! Output: [real(r8) (:)   ]  CO2 compensation point (Pa)
            kc         => photosyns_inst%kc_patch               , & ! Output: [real(r8) (:)   ]  Michaelis-Menten constant for CO2 (Pa)
            ko         => photosyns_inst%ko_patch               , & ! Output: [real(r8) (:)   ]  Michaelis-Menten constant for O2 (Pa)
            qe         => photosyns_inst%qe_patch               , & ! Output: [real(r8) (:)   ]  quantum efficiency, used only for C4 (mol CO2 / mol photons)
            tpu_z      => photosyns_inst%tpu_z_patch            , & ! Output: [real(r8) (:,:) ]  triose phosphate utilization rate (umol CO2/m**2/s)
            kp_z       => photosyns_inst%kp_z_patch             , & ! Output: [real(r8) (:,:) ]  initial slope of CO2 response curve (C4 plants)
            bbb        => photosyns_inst%bbb_patch              , & ! Output: [real(r8) (:)   ]  Ball-Berry minimum leaf conductance (umol H2O/m**2/s)
            mbb        => photosyns_inst%mbb_patch              , & ! Output: [real(r8) (:)   ]  Ball-Berry slope of conductance-photosynthesis relationship
            rh_leaf    => photosyns_inst%rh_leaf_patch          , & ! Output: [real(r8) (:)   ]  fractional humidity at leaf surface (dimensionless)
            vpd_can    => photosyns_inst%vpd_can_patch          , & ! Output: [real(r8) (:)   ]  canopy vapor pressure deficit (kPa)
            lnc        => photosyns_inst%lnca_patch             , & ! Output: [real(r8) (:)   ]  top leaf layer leaf N concentration (gN leaf/m^2)
            light_inhibit=> photosyns_inst%light_inhibit        , & ! Input:  [logical        ]  flag if light should inhibit respiration
            leafresp_method=> photosyns_inst%leafresp_method    , & ! Input:  [integer        ]  method type to use for leaf-maint.-respiration at 25C canopy top
            medlynintercept  => pftcon%medlynintercept          , & ! Input:  [real(r8) (:)   ]  Intercept for Medlyn stomatal conductance model method
            stomatalcond_mtd=> photosyns_inst%stomatalcond_mtd  , & ! Input:  [integer        ]  method type to use for stomatal conductance.GC.fnlprmsn15_r22845
            leaf_mr_vcm => canopystate_inst%leaf_mr_vcm           & ! Input:  [real(r8)       ]  scalar constant of leaf respiration with Vcmax
            )

        if (phase == 'sun') then
            par_z     =>    solarabs_inst%parsun_z_patch        ! Input:  [real(r8) (:,:) ]  par absorbed per unit lai for canopy layer (w/m**2)
            lai_z     =>    canopystate_inst%laisun_z_patch     ! Input:  [real(r8) (:,:) ]  leaf area index for canopy layer, sunlit or shaded
            vcmaxcint =>    surfalb_inst%vcmaxcintsun_patch     ! Input:  [real(r8) (:)   ]  leaf to canopy scaling coefficient
            alphapsn  =>    photosyns_inst%alphapsnsun_patch    ! Input:  [real(r8) (:)   ]  13C fractionation factor for PSN ()
            o3coefv   =>    ozone_inst%o3coefvsun_patch         ! Input:  [real(r8) (:)   ]  O3 coefficient used in photosynthesis calculation
            o3coefg   =>    ozone_inst%o3coefgsun_patch         ! Input:  [real(r8) (:)   ]  O3 coefficient used in rs calculation
            ci_z      =>    photosyns_inst%cisun_z_patch        ! Output: [real(r8) (:,:) ]  intracellular leaf CO2 (Pa)
            rs        =>    photosyns_inst%rssun_patch          ! Output: [real(r8) (:)   ]  leaf stomatal resistance (s/m)
            rs_z      =>    photosyns_inst%rssun_z_patch        ! Output: [real(r8) (:,:) ]  canopy layer: leaf stomatal resistance (s/m)
            lmr       =>    photosyns_inst%lmrsun_patch         ! Output: [real(r8) (:)   ]  leaf maintenance respiration rate (umol CO2/m**2/s)
            lmr_z     =>    photosyns_inst%lmrsun_z_patch       ! Output: [real(r8) (:,:) ]  canopy layer: leaf maintenance respiration rate (umol CO2/m**2/s)
            psn       =>    photosyns_inst%psnsun_patch         ! Output: [real(r8) (:)   ]  foliage photosynthesis (umol co2 /m**2/ s) [always +]
            psn_z     =>    photosyns_inst%psnsun_z_patch       ! Output: [real(r8) (:,:) ]  canopy layer: foliage photosynthesis (umol co2 /m**2/ s) [always +]
            psn_wc    =>    photosyns_inst%psnsun_wc_patch      ! Output: [real(r8) (:)   ]  Rubisco-limited foliage photosynthesis (umol co2 /m**2/ s) [always +]
            psn_wj    =>    photosyns_inst%psnsun_wj_patch      ! Output: [real(r8) (:)   ]  RuBP-limited foliage photosynthesis (umol co2 /m**2/ s) [always +]
            psn_wp    =>    photosyns_inst%psnsun_wp_patch      ! Output: [real(r8) (:)   ]  product-limited foliage photosynthesis (umol co2 /m**2/ s) [always +]
        else if (phase == 'sha') then
            par_z     =>    solarabs_inst%parsha_z_patch        ! Input:  [real(r8) (:,:) ]  par absorbed per unit lai for canopy layer (w/m**2)
            lai_z     =>    canopystate_inst%laisha_z_patch     ! Input:  [real(r8) (:,:) ]  leaf area index for canopy layer, sunlit or shaded
            vcmaxcint =>    surfalb_inst%vcmaxcintsha_patch     ! Input:  [real(r8) (:)   ]  leaf to canopy scaling coefficient
            alphapsn  =>    photosyns_inst%alphapsnsha_patch    ! Input:  [real(r8) (:)   ]  13C fractionation factor for PSN ()
            o3coefv   =>    ozone_inst%o3coefvsha_patch         ! Input:  [real(r8) (:)   ]  O3 coefficient used in photosynthesis calculation
            o3coefg   =>    ozone_inst%o3coefgsha_patch         ! Input:  [real(r8) (:)   ]  O3 coefficient used in rs calculation
            ci_z      =>    photosyns_inst%cisha_z_patch        ! Output: [real(r8) (:,:) ]  intracellular leaf CO2 (Pa)
            rs        =>    photosyns_inst%rssha_patch          ! Output: [real(r8) (:)   ]  leaf stomatal resistance (s/m)
            rs_z      =>    photosyns_inst%rssha_z_patch        ! Output: [real(r8) (:,:) ]  canopy layer: leaf stomatal resistance (s/m)
            lmr       =>    photosyns_inst%lmrsha_patch         ! Output: [real(r8) (:)   ]  leaf maintenance respiration rate (umol CO2/m**2/s)
            lmr_z     =>    photosyns_inst%lmrsha_z_patch       ! Output: [real(r8) (:,:) ]  canopy layer: leaf maintenance respiration rate (umol CO2/m**2/s)
            psn       =>    photosyns_inst%psnsha_patch         ! Output: [real(r8) (:)   ]  foliage photosynthesis (umol co2 /m**2/ s) [always +]
            psn_z     =>    photosyns_inst%psnsha_z_patch       ! Output: [real(r8) (:,:) ]  canopy layer: foliage photosynthesis (umol co2 /m**2/ s) [always +]
            psn_wc    =>    photosyns_inst%psnsha_wc_patch      ! Output: [real(r8) (:)   ]  Rubisco-limited foliage photosynthesis (umol co2 /m**2/ s) [always +]
            psn_wj    =>    photosyns_inst%psnsha_wj_patch      ! Output: [real(r8) (:)   ]  RuBP-limited foliage photosynthesis (umol co2 /m**2/ s) [always +]
            psn_wp    =>    photosyns_inst%psnsha_wp_patch      ! Output: [real(r8) (:)   ]  product-limited foliage photosynthesis (umol co2 /m**2/ s) [always +]
        end if

        !==============================================================================!
        ! Photosynthesis and stomatal conductance parameters, from:
        ! Bonan et al (2011) JGR, 116, doi:10.1029/2010JG001593
        !==============================================================================!

        ! Determine seconds of current time step

        dtime = get_step_size_real()

        ! Activation energy, from:
        ! Bernacchi et al (2001) Plant, Cell and Environment 24:253-259
        ! Bernacchi et al (2003) Plant, Cell and Environment 26:1419-1430
        ! except TPU from: Harley et al (1992) Plant, Cell and Environment 15:271-282

        ! High temperature deactivation, from:
        ! Leuning (2002) Plant, Cell and Environment 25:1205-1210
        ! The factor "c" scales the deactivation to a value of 1.0 at 25C

        lmrc    = fth25 (params_inst%lmrhd, params_inst%lmrse)

        ! Miscellaneous parameters, from Bonan et al (2011) JGR, 116, doi:10.1029/2010JG001593

        do f = 1, fn
            p = filterp(f)
            c = patch%column(p)

            ! C3 or C4 photosynthesis logical variable

            if (nint(c3psn(patch%itype(p))) == 1) then
            c3flag(p) = .true.
            else if (nint(c3psn(patch%itype(p))) == 0) then
            c3flag(p) = .false.
            end if

            ! C3 and C4 dependent parameters

            if (c3flag(p)) then
            qe(p) = 0._r8
            if ( stomatalcond_mtd == stomatalcond_mtd_bb1987 ) bbbopt(p) = bbbopt_c3
            else
            qe(p) = 0.05_r8
            if ( stomatalcond_mtd == stomatalcond_mtd_bb1987 ) bbbopt(p) = bbbopt_c4
            end if

            ! Soil water stress applied to Ball-Berry parameters

            if ( stomatalcond_mtd == stomatalcond_mtd_bb1987 ) then
            bbb(p) = max (bbbopt(p)*btran(p), 1._r8)
            mbb(p) = mbbopt(patch%itype(p))
            end if

            ! kc, ko, cp, from: Bernacchi et al (2001) Plant, Cell and Environment 24:253-259
            !
            !       kc25_coef = 404.9e-6 mol/mol
            !       ko25_coef = 278.4e-3 mol/mol
            !       cp25_yr2000 = 42.75e-6 mol/mol
            !
            ! Derive sco from cp and O2 using present-day O2 (0.209 mol/mol) and re-calculate
            ! cp to account for variation in O2 using cp = 0.5 O2 / sco
            !

            kc25 = params_inst%kc25_coef * forc_pbot(c)
            ko25 = params_inst%ko25_coef * forc_pbot(c)
            sco  = 0.5_r8 * 0.209_r8 / params_inst%cp25_yr2000
            cp25 = 0.5_r8 * oair(p) / sco

            kc(p) = kc25 * ft(t_veg(p), params_inst%kcha)
            ko(p) = ko25 * ft(t_veg(p), params_inst%koha)
            cp(p) = cp25 * ft(t_veg(p), params_inst%cpha)

        end do

        ! Multi-layer parameters scaled by leaf nitrogen profile.
        ! Loop through each canopy layer to calculate nitrogen profile using
        ! cumulative lai at the midpoint of the layer

        do f = 1, fn
            p = filterp(f)

            if (lnc_opt .eqv. .false.) then     
            ! Leaf nitrogen concentration at the top of the canopy (g N leaf / m**2 leaf)
            
            if ( (slatop(patch%itype(p)) *leafcn(patch%itype(p))) .le. 0.0_r8)then
                call endrun(subgrid_index=p, subgrid_level=subgrid_level_patch, msg="ERROR: slatop or leafcn is zero")
            end if
            lnc(p) = 1._r8 / (slatop(patch%itype(p)) * leafcn(patch%itype(p)))
            end if   

            ! Using the actual nitrogen allocated to the leaf after
            ! uptake rather than fixing leaf nitrogen based on SLA and CN
            ! ratio
            if (lnc_opt .eqv. .true.) then                                                     
            ! nlevcan and nrad(p) look like the same variable ?? check this later
            sum_nscaler = 0.0_r8                                                    
            laican = 0.0_r8                                                         
            total_lai = 0.0_r8                                                      

            do iv = 1, nrad(p)                                                      

                if (iv == 1) then                                                    
                    laican = 0.5_r8 * tlai_z(p,iv)                                    
                    total_lai = tlai_z(p,iv)                                          
                else                                                                 
                    laican = laican + 0.5_r8 * (tlai_z(p,iv-1)+tlai_z(p,iv))          
                    total_lai = total_lai + tlai_z(p,iv)                              
                end if                                                               

                ! Scale for leaf nitrogen profile. If multi-layer code, use explicit
                ! profile. If sun/shade big leaf code, use canopy integrated factor.
                if (nlevcan == 1) then                                               
                    nscaler = 1.0_r8                                                  
                else if (nlevcan > 1) then                                           
                    nscaler = exp(-kn(p) * laican)                                    
                end if                                                               

                sum_nscaler = sum_nscaler + nscaler                                  

            end do                                                                  

            if (tlai(p) > 0.0_r8 .AND. sum_nscaler > 0.0_r8) then
                ! dividing by LAI to convert total leaf nitrogen
                ! from m2 ground to m2 leaf; dividing by sum_nscaler to
                ! convert total leaf N to leaf N at canopy top
                lnc(p) = leafn(p) / (tlai(p) * sum_nscaler)
            else                                                                    
                lnc(p) = 0.0_r8                                                      
            end if                                                                  

            end if                                                                     


            ! reduce_dayl_factor .eqv. .false.  
            if (reduce_dayl_factor .eqv. .true.) then                                          
            if (dayl_factor(p) > 0.25_r8) then
                ! dayl_factor(p) = 1.0_r8  
            end if                                                                  
            end if                                                                     


            ! Default
            if (vcmax_opt == 0) then                                                   
            ! vcmax25 at canopy top, as in CN but using lnc at top of the canopy
            vcmax25top = lnc(p) * flnr(patch%itype(p)) * params_inst%fnr * params_inst%act25 * dayl_factor(p)
            if (.not. use_cn) then
                vcmax25top = vcmax25top * fnitr(patch%itype(p))
            else
                if ( CNAllocate_Carbon_only() ) vcmax25top = vcmax25top * fnitr(patch%itype(p))
            end if
            else if (vcmax_opt == 3) then                                                                   
            vcmax25top = ( i_vcad(patch%itype(p)) + s_vcad(patch%itype(p)) * lnc(p) ) * dayl_factor(p)  
            else if (vcmax_opt == 4) then                                                                   
            nptreemax = 9  ! is this number correct? check later 
            if (patch%itype(p) >= nptreemax) then   ! if not tree 
                ! for shrubs and herbs 
                vcmax25top = lnc(p) * ( i_flnr(patch%itype(p)) + s_flnr(patch%itype(p)) * lnc(p) ) * params_inst%fnr * params_inst%act25 * &
                    dayl_factor(p)
            else
                ! if tree 
                vcmax25top = lnc(p) * ( i_flnr(patch%itype(p)) * exp(s_flnr(patch%itype(p)) * lnc(p)) ) * params_inst%fnr * params_inst%act25 * &
                    dayl_factor(p)
                ! for trees 
            end if     
            end if        


            ! Parameters derived from vcmax25top. Bonan et al (2011) JGR, 116, doi:10.1029/2010JG001593
            ! used jmax25 = 1.97 vcmax25, from Wullschleger (1993) Journal of Experimental Botany 44:907-920.

            jmax25top = ((2.59_r8 - 0.035_r8*min(max((t10(p)-tfrz),11._r8),35._r8)) * vcmax25top) * &
                            params_inst%jmax25top_sf
            tpu25top  = params_inst%tpu25ratio * vcmax25top
            kp25top   = params_inst%kp25ratio * vcmax25top

            ! Nitrogen scaling factor. Bonan et al (2011) JGR, 116, doi:10.1029/2010JG001593 used
            ! kn = 0.11. Here, derive kn from vcmax25 as in Lloyd et al (2010) Biogeosciences, 7, 1833-1859
            ! Remove daylength factor from vcmax25 so that kn is based on maximum vcmax25
            ! But not used as defined here if using sun/shade big leaf code. Instead,
            ! will use canopy integrated scaling factors from SurfaceAlbedo.

            if (dayl_factor(p)  < 1.0e-12_r8) then
            kn(p) =  0._r8
            else
            kn(p) = exp(0.00963_r8 * vcmax25top/dayl_factor(p) - 2.43_r8)
            end if

            if (use_cn) then
            if ( leafresp_method == leafresp_mtd_ryan1991 ) then
            ! Leaf maintenance respiration to match the base rate used in CN
            ! but with the new temperature functions for C3 and C4 plants.
            !
            ! Base rate for maintenance respiration is from:
            ! M. Ryan, 1991. Effects of climate change on plant respiration.
            ! Ecological Applications, 1(2), 157-167.
            ! Original expression is br = 0.0106 molC/(molN h)
            ! Conversion by molecular weights of C and N gives 2.525e-6 gC/(gN s)
            !
            ! Base rate is at 20C. Adjust to 25C using the CN Q10 = 1.5
            !
            ! CN respiration has units:  g C / g N [leaf] / s. This needs to be
            ! converted from g C / g N [leaf] / s to umol CO2 / m**2 [leaf] / s
            !
            ! Then scale this value at the top of the canopy for canopy depth

                lmr25top = 2.525e-6_r8 * (1.5_r8 ** ((25._r8 - 20._r8)/10._r8))
                lmr25top = lmr25top * lnc(p) / 12.e-06_r8
            
            else if ( leafresp_method == leafresp_mtd_atkin2015 ) then
                !using new form for respiration base rate from Atkin
                !communication. 
                if ( lnc(p) > 0.0_r8 ) then
                    lmr25top = params_inst%lmr_intercept_atkin(ivt(p)) + (lnc(p) * 0.2061_r8) - (0.0402_r8 * (t10(p)-tfrz))
                else
                    lmr25top = 0.0_r8
                end if
            end if

            else
            ! Leaf maintenance respiration in proportion to vcmax25top

            if (c3flag(p)) then
                lmr25top = vcmax25top * leaf_mr_vcm
            else
                lmr25top = vcmax25top * 0.025_r8
            end if
            end if

            ! Loop through canopy layers (above snow). Respiration needs to be
            ! calculated every timestep. Others are calculated only if daytime

            laican = 0._r8
            do iv = 1, nrad(p)

            ! Cumulative lai at middle of layer

            if (iv == 1) then
                laican = 0.5_r8 * tlai_z(p,iv)
            else
                laican = laican + 0.5_r8 * (tlai_z(p,iv-1)+tlai_z(p,iv))
            end if

            ! Scale for leaf nitrogen profile. If multi-layer code, use explicit
            ! profile. If sun/shade big leaf code, use canopy integrated factor.

            if (nlevcan == 1) then
                nscaler = vcmaxcint(p)
            else if (nlevcan > 1) then
                nscaler = exp(-kn(p) * laican)
            end if

            ! Maintenance respiration

            lmr25 = lmr25top * nscaler

            if(use_luna.and.c3flag(p).and.crop(patch%itype(p))== 0) then
                if(.not.use_cn)then ! If CN is on, use leaf N to predict respiration (above). Otherwise, use Vcmax term from LUNA.  RF
                    lmr25 = leaf_mr_vcm * photosyns_inst%vcmx25_z_patch(p,iv)
                endif
            endif
            
            if (c3flag(p)) then
                lmr_z(p,iv) = lmr25 * ft(t_veg(p), params_inst%lmrha) * fth(t_veg(p), params_inst%lmrhd, &
                    params_inst%lmrse, lmrc)
            else
                lmr_z(p,iv) = lmr25 * 2._r8**((t_veg(p)-(tfrz+25._r8))/10._r8)
                lmr_z(p,iv) = lmr_z(p,iv) / (1._r8 + exp( 1.3_r8*(t_veg(p)-(tfrz+55._r8)) ))
            end if

            if (par_z(p,iv) <= 0._r8) then           ! night time

                vcmax_z(p,iv) = 0._r8
                jmax_z(p,iv) = 0._r8
                tpu_z(p,iv) = 0._r8
                kp_z(p,iv) = 0._r8

                if ( use_c13 ) then
                    alphapsn(p) = 1._r8
                end if

            else                                     ! day time

                if(use_luna.and.c3flag(p).and.crop(patch%itype(p))== 0)then
                    vcmax25 = photosyns_inst%vcmx25_z_patch(p,iv)
                    jmax25 = photosyns_inst%jmx25_z_patch(p,iv)
                    tpu25 = params_inst%tpu25ratio * vcmax25 
                    !Implement scaling of Vcmax25 from sunlit average to shaded canopy average value. RF & GBB. 1 July 2016
                    if(phase == 'sha'.and.surfalb_inst%vcmaxcintsun_patch(p).gt.0._r8.and.nlevcan==1) then
                        vcmax25 = vcmax25 * surfalb_inst%vcmaxcintsha_patch(p)/surfalb_inst%vcmaxcintsun_patch(p)
                        jmax25  = jmax25  * surfalb_inst%vcmaxcintsha_patch(p)/surfalb_inst%vcmaxcintsun_patch(p) 
                        tpu25   = tpu25   * surfalb_inst%vcmaxcintsha_patch(p)/surfalb_inst%vcmaxcintsun_patch(p) 
                    end if
                            
                else
                    vcmax25 = vcmax25top * nscaler
                    jmax25 = jmax25top * nscaler
                    tpu25 = tpu25top * nscaler                  
                endif
                kp25 = kp25top * nscaler

                ! Adjust for temperature

                vcmaxse = (668.39_r8 - 1.07_r8 * min(max((t10(p)-tfrz),11._r8),35._r8)) * params_inst%vcmaxse_sf
                jmaxse  = (659.70_r8 - 0.75_r8 * min(max((t10(p)-tfrz),11._r8),35._r8)) * params_inst%jmaxse_sf
                tpuse   = (668.39_r8 - 1.07_r8 * min(max((t10(p)-tfrz),11._r8),35._r8)) * params_inst%tpuse_sf
                vcmaxc = fth25 (params_inst%vcmaxhd, vcmaxse)
                jmaxc  = fth25 (params_inst%jmaxhd, jmaxse)
                tpuc   = fth25 (params_inst%tpuhd, tpuse)
                vcmax_z(p,iv) = vcmax25 * ft(t_veg(p), params_inst%vcmaxha) * fth(t_veg(p), &
                    params_inst%vcmaxhd, vcmaxse, vcmaxc)
                jmax_z(p,iv) = jmax25 * ft(t_veg(p), params_inst%jmaxha) * fth(t_veg(p), &
                    params_inst%jmaxhd, jmaxse, jmaxc)
                tpu_z(p,iv) = tpu25 * ft(t_veg(p), params_inst%tpuha) * fth(t_veg(p), params_inst%tpuhd, tpuse, tpuc)

                if (.not. c3flag(p)) then
                    vcmax_z(p,iv) = vcmax25 * 2._r8**((t_veg(p)-(tfrz+25._r8))/10._r8)
                    vcmax_z(p,iv) = vcmax_z(p,iv) / (1._r8 + exp( 0.2_r8*((tfrz+15._r8)-t_veg(p)) ))
                    vcmax_z(p,iv) = vcmax_z(p,iv) / (1._r8 + exp( 0.3_r8*(t_veg(p)-(tfrz+40._r8)) ))
                end if

                kp_z(p,iv) = kp25 * 2._r8**((t_veg(p)-(tfrz+25._r8))/10._r8)

            end if

            ! Adjust for soil water

            vcmax_z(p,iv) = vcmax_z(p,iv) * btran(p)
            lmr_z(p,iv) = lmr_z(p,iv) * btran(p)
            
            ! Change to add in light inhibition of respiration. 0.67 from Lloyd et al. 2010, & Metcalfe et al. 2012 
            ! Also pers. comm from Peter Reich (Nov 2015). Might potentially be updated pending findings of Atkin et al. (in prep)
            ! review of light inhibition database. 
            if ( light_inhibit .and. par_z(p,1) > 0._r8) then ! are the lights on? 
                lmr_z(p,iv) = lmr_z(p,iv) * 0.67_r8 ! inhibit respiration accordingly. 
            end if

            end do       ! canopy layer loop
        end do          ! patch loop

        !==============================================================================!
        ! Leaf-level photosynthesis and stomatal conductance
        !==============================================================================!

        rsmax0 = 2.e4_r8

        do f = 1, fn
            p = filterp(f)
            c = patch%column(p)
            g = patch%gridcell(p)

            ! Leaf boundary layer conductance, umol/m**2/s

            cf = forc_pbot(c)/(rgas*1.e-3_r8*tgcm(p))*1.e06_r8
            gb = 1._r8/rb(p)
            gb_mol(p) = gb * cf

            ! Loop through canopy layers (above snow). Only do calculations if daytime

            do iv = 1, nrad(p)

            if (par_z(p,iv) <= 0._r8) then           ! night time

                ac(p,iv) = 0._r8
                aj(p,iv) = 0._r8
                ap(p,iv) = 0._r8
                ag(p,iv) = 0._r8
                an(p,iv) = ag(p,iv) - lmr_z(p,iv)
                psn_z(p,iv) = 0._r8
                psn_wc_z(p,iv) = 0._r8
                psn_wj_z(p,iv) = 0._r8
                psn_wp_z(p,iv) = 0._r8
                if (      stomatalcond_mtd == stomatalcond_mtd_bb1987 )then
                    rs_z(p,iv) = min(rsmax0, 1._r8/bbb(p) * cf)
                else if ( stomatalcond_mtd == stomatalcond_mtd_medlyn2011 )then
                    rs_z(p,iv) = min(rsmax0, 1._r8/medlynintercept(patch%itype(p)) * cf)
                end if
                ci_z(p,iv) = 0._r8
                rh_leaf(p) = 0._r8

            else                                     ! day time

                !now the constraint is no longer needed, Jinyun Tang
                ceair = min( eair(p),  esat_tv(p) )
                if (      stomatalcond_mtd == stomatalcond_mtd_bb1987 )then
                    rh_can = ceair / esat_tv(p)
                else if ( stomatalcond_mtd == stomatalcond_mtd_medlyn2011 )then
                    ! Put some constraints on RH in the canopy when Medlyn stomatal conductance is being used
                    rh_can = max((esat_tv(p) - ceair), medlyn_rh_can_max) * medlyn_rh_can_fact
                    vpd_can(p) = rh_can
                end if

                ! Electron transport rate for C3 plants. Convert par from W/m2 to
                ! umol photons/m**2/s using the factor 4.6

                qabs  = 0.5_r8 * (1._r8 - params_inst%fnps) * par_z(p,iv) * 4.6_r8
                aquad = params_inst%theta_psii
                bquad = -(qabs + jmax_z(p,iv))
                cquad = qabs * jmax_z(p,iv)
                call quadratic (aquad, bquad, cquad, r1, r2)
                je = min(r1,r2)

                ! Iterative loop for ci beginning with initial guess

                if (c3flag(p)) then
                    ci_z(p,iv) = 0.7_r8 * cair(p)
                else
                    ci_z(p,iv) = 0.4_r8 * cair(p)
                end if

                niter = 0

                ! Increment iteration counter. Stop if too many iterations

                niter = niter + 1

                ! Save old ci

                ciold = ci_z(p,iv)

                !find ci and stomatal conductance
                call hybrid(ciold, p, iv, c, gb_mol(p), je, cair(p), oair(p), &
                    lmr_z(p,iv), par_z(p,iv), rh_can, gs_mol(p,iv), niter, &
                    atm2lnd_inst, photosyns_inst)

                ! End of ci iteration.  Check for an < 0, in which case gs_mol = bbb

                if (an(p,iv) < 0._r8) then
                    if (stomatalcond_mtd == stomatalcond_mtd_bb1987) then
                        gs_mol(p,iv) = bbb(p)
                    else if ( stomatalcond_mtd == stomatalcond_mtd_medlyn2011 )then
                        gs_mol(p,iv) = medlynintercept(patch%itype(p))
                    end if
                end if

                !
                ! This sets the  variables GSSUN and GSSHA
                !
                ! Write stomatal conductance to the appropriate phase
                if (phase=='sun') then
                    gs_mol_sun(p,iv) = gs_mol(p,iv)
                else if (phase=='sha') then
                    gs_mol_sha(p,iv) = gs_mol(p,iv)
                end if

                ! Use time period 1 hour before and 1 hour after local noon inclusive (11AM-1PM)
                if ( is_near_local_noon( grc%londeg(g), deltasec=3600 ) )then
                    if (phase == 'sun') then
                        gs_mol_sun_ln(p,iv) = gs_mol(p,iv)
                    else if (phase == 'sha') then
                        gs_mol_sha_ln(p,iv) = gs_mol(p,iv)
                    end if
                else
                    if (phase == 'sun') then
                        gs_mol_sun_ln(p,iv) = spval
                    else if (phase == 'sha') then
                        gs_mol_sha_ln(p,iv) = spval
                    end if
                end if

                ! Final estimates for cs and ci (needed for early exit of ci iteration when an < 0)

                cs = cair(p) - 1.4_r8/gb_mol(p) * an(p,iv) * forc_pbot(c)
                cs = max(cs,max_cs)
                ci_z(p,iv) = cair(p) - an(p,iv) * forc_pbot(c) * (1.4_r8*gs_mol(p,iv)+1.6_r8*gb_mol(p)) / (gb_mol(p)*gs_mol(p,iv))

                ! Trap for values of ci_z less than 1.e-06.  This is needed for
                ! Megan (which can crash with negative values)
                ci_z(p,iv) = max( ci_z(p,iv), 1.e-06_r8 )

                ! Convert gs_mol (umol H2O/m**2/s) to gs (m/s) and then to rs (s/m)

                gs = gs_mol(p,iv) / cf
                rs_z(p,iv) = min(1._r8/gs, rsmax0)
                rs_z(p,iv) = rs_z(p,iv) / o3coefg(p)

                ! Photosynthesis. Save rate-limiting photosynthesis

                psn_z(p,iv) = ag(p,iv)
                psn_z(p,iv) = psn_z(p,iv) * o3coefv(p)

                psn_wc_z(p,iv) = 0._r8
                psn_wj_z(p,iv) = 0._r8
                psn_wp_z(p,iv) = 0._r8

                if (ac(p,iv) <= aj(p,iv) .and. ac(p,iv) <= ap(p,iv)) then
                    psn_wc_z(p,iv) =  psn_z(p,iv)
                else if (aj(p,iv) < ac(p,iv) .and. aj(p,iv) <= ap(p,iv)) then
                    psn_wj_z(p,iv) =  psn_z(p,iv)
                else if (ap(p,iv) < ac(p,iv) .and. ap(p,iv) < aj(p,iv)) then
                    psn_wp_z(p,iv) =  psn_z(p,iv)
                end if

                ! Make sure iterative solution is correct

                if (gs_mol(p,iv) < 0._r8) then
                    write (iulog,*)'Negative stomatal conductance:'
                    write (iulog,*)'p,iv,gs_mol= ',p,iv,gs_mol(p,iv)
                    call endrun(subgrid_index=p, subgrid_level=subgrid_level_patch, msg=errmsg(sourcefile, 1963))
                end if

                ! Compare with Ball-Berry model: gs_mol = m * an * hs/cs p + b
                if ( stomatalcond_mtd == stomatalcond_mtd_bb1987 )then
                    hs = (gb_mol(p)*ceair + gs_mol(p,iv)*esat_tv(p)) / ((gb_mol(p)+gs_mol(p,iv))*esat_tv(p))
                    rh_leaf(p) = hs
                    gs_mol_err = mbb(p)*max(an(p,iv), 0._r8)*hs/cs*forc_pbot(c) + bbb(p)
                    
                    if (abs(gs_mol(p,iv)-gs_mol_err) > 1.e-01_r8) then
                        write (iulog,*) 'Ball-Berry error check - stomatal conductance error:'
                        write (iulog,*) gs_mol(p,iv), gs_mol_err
                    end if
                endif
            end if    ! night or day if branch
            end do       ! canopy layer loop
        end do          ! patch loop

        !==============================================================================!
        ! Canopy photosynthesis and stomatal conductance
        !==============================================================================!

        ! Sum canopy layer fluxes and then derive effective leaf-level fluxes (per
        ! unit leaf area), which are used in other parts of the model. Here, laican
        ! sums to either laisun or laisha.

        do f = 1, fn
            p = filterp(f)

            psncan = 0._r8
            psncan_wc = 0._r8
            psncan_wj = 0._r8
            psncan_wp = 0._r8
            lmrcan = 0._r8
            gscan = 0._r8
            laican = 0._r8
            do iv = 1, nrad(p)
            psncan = psncan + psn_z(p,iv) * lai_z(p,iv)
            psncan_wc = psncan_wc + psn_wc_z(p,iv) * lai_z(p,iv)
            psncan_wj = psncan_wj + psn_wj_z(p,iv) * lai_z(p,iv)
            psncan_wp = psncan_wp + psn_wp_z(p,iv) * lai_z(p,iv)
            lmrcan = lmrcan + lmr_z(p,iv) * lai_z(p,iv)
            gscan = gscan + lai_z(p,iv) / (rb(p)+rs_z(p,iv))
            laican = laican + lai_z(p,iv)
            end do
            if (laican > 0._r8) then
            psn(p) = psncan / laican
            psn_wc(p) = psncan_wc / laican
            psn_wj(p) = psncan_wj / laican
            psn_wp(p) = psncan_wp / laican
            lmr(p) = lmrcan / laican
            rs(p) = laican / gscan - rb(p)
            else
            psn(p) =  0._r8
            psn_wc(p) =  0._r8
            psn_wj(p) =  0._r8
            psn_wp(p) =  0._r8
            lmr(p) = 0._r8
            rs(p) = 0._r8
            end if
        end do

    end associate

    end subroutine Photosynthesis

    !------------------------------------------------------------------------------
    subroutine PhotosynthesisTotal (fn, filterp, &
        atm2lnd_inst, canopystate_inst, photosyns_inst)
    !
    ! Determine total photosynthesis
    !
    ! !ARGUMENTS:
    integer                , intent(in)    :: fn                             ! size of pft filter
    integer                , intent(in)    :: filterp(fn)                    ! patch filter
    type(atm2lnd_type)     , intent(in)    :: atm2lnd_inst
    type(canopystate_type) , intent(in)    :: canopystate_inst
    type(photosyns_type)   , intent(inout) :: photosyns_inst
    !
    ! !LOCAL VARIABLES:
    integer :: f,fp,p,l,g               ! indices

    real(r8) :: rc14_atm(nsectors_c14), rc13_atm
    integer :: sector_c14
    !-----------------------------------------------------------------------

    associate(                                             &
            forc_pco2   => atm2lnd_inst%forc_pco2_grc       , & ! Input:  [real(r8) (:) ]  partial pressure co2 (Pa)
            forc_pc13o2 => atm2lnd_inst%forc_pc13o2_grc     , & ! Input:  [real(r8) (:) ]  partial pressure c13o2 (Pa)
            forc_po2    => atm2lnd_inst%forc_po2_grc        , & ! Input:  [real(r8) (:) ]  partial pressure o2 (Pa)

            laisun      => canopystate_inst%laisun_patch    , & ! Input:  [real(r8) (:) ]  sunlit leaf area
            laisha      => canopystate_inst%laisha_patch    , & ! Input:  [real(r8) (:) ]  shaded leaf area

            psnsun      => photosyns_inst%psnsun_patch      , & ! Input:  [real(r8) (:) ]  sunlit leaf photosynthesis (umol CO2 /m**2/ s)
            psnsha      => photosyns_inst%psnsha_patch      , & ! Input:  [real(r8) (:) ]  shaded leaf photosynthesis (umol CO2 /m**2/ s)
            rc13_canair => photosyns_inst%rc13_canair_patch , & ! Output: [real(r8) (:) ]  C13O2/C12O2 in canopy air
            rc13_psnsun => photosyns_inst%rc13_psnsun_patch , & ! Output: [real(r8) (:) ]  C13O2/C12O2 in sunlit canopy psn flux
            rc13_psnsha => photosyns_inst%rc13_psnsha_patch , & ! Output: [real(r8) (:) ]  C13O2/C12O2 in shaded canopy psn flux
            alphapsnsun => photosyns_inst%alphapsnsun_patch , & ! Output: [real(r8) (:) ]  fractionation factor in sunlit canopy psn flux
            alphapsnsha => photosyns_inst%alphapsnsha_patch , & ! Output: [real(r8) (:) ]  fractionation factor in shaded canopy psn flux
            psnsun_wc   => photosyns_inst%psnsun_wc_patch   , & ! Output: [real(r8) (:) ]  Rubsico-limited sunlit leaf photosynthesis (umol CO2 /m**2/ s)
            psnsun_wj   => photosyns_inst%psnsun_wj_patch   , & ! Output: [real(r8) (:) ]  RuBP-limited sunlit leaf photosynthesis (umol CO2 /m**2/ s)
            psnsun_wp   => photosyns_inst%psnsun_wp_patch   , & ! Output: [real(r8) (:) ]  product-limited sunlit leaf photosynthesis (umol CO2 /m**2/ s)
            psnsha_wc   => photosyns_inst%psnsha_wc_patch   , & ! Output: [real(r8) (:) ]  Rubsico-limited shaded leaf photosynthesis (umol CO2 /m**2/ s)
            psnsha_wj   => photosyns_inst%psnsha_wj_patch   , & ! Output: [real(r8) (:) ]  RuBP-limited shaded leaf photosynthesis (umol CO2 /m**2/ s)
            psnsha_wp   => photosyns_inst%psnsha_wp_patch   , & ! Output: [real(r8) (:) ]  product-limited shaded leaf photosynthesis (umol CO2 /m**2/ s)
            c13_psnsun  => photosyns_inst%c13_psnsun_patch  , & ! Output: [real(r8) (:) ]  sunlit leaf photosynthesis (umol 13CO2 /m**2/ s)
            c13_psnsha  => photosyns_inst%c13_psnsha_patch  , & ! Output: [real(r8) (:) ]  shaded leaf photosynthesis (umol 13CO2 /m**2/ s)
            c14_psnsun  => photosyns_inst%c14_psnsun_patch  , & ! Output: [real(r8) (:) ]  sunlit leaf photosynthesis (umol 14CO2 /m**2/ s)
            c14_psnsha  => photosyns_inst%c14_psnsha_patch  , & ! Output: [real(r8) (:) ]  shaded leaf photosynthesis (umol 14CO2 /m**2/ s)
            fpsn        => photosyns_inst%fpsn_patch        , & ! Output: [real(r8) (:) ]  photosynthesis (umol CO2 /m**2 /s)
            fpsn_wc     => photosyns_inst%fpsn_wc_patch     , & ! Output: [real(r8) (:) ]  Rubisco-limited photosynthesis (umol CO2 /m**2 /s)
            fpsn_wj     => photosyns_inst%fpsn_wj_patch     , & ! Output: [real(r8) (:) ]  RuBP-limited photosynthesis (umol CO2 /m**2 /s)
            fpsn_wp     => photosyns_inst%fpsn_wp_patch       & ! Output: [real(r8) (:) ]  product-limited photosynthesis (umol CO2 /m**2 /s)
            )

        if ( use_c14 ) then
            if (use_c14_bombspike) then
            call C14BombSpike(rc14_atm)
            else
            rc14_atm(:) = c14ratio
            end if
        end if

        if ( use_c13 ) then
            if (use_c13_timeseries) then
            call C13TimeSeries(rc13_atm)
            end if
        end if

        do f = 1, fn
            p = filterp(f)
            g = patch%gridcell(p)

            if (.not. use_fates) then
            fpsn(p)    = psnsun(p)   *laisun(p) + psnsha(p)   *laisha(p)
            fpsn_wc(p) = psnsun_wc(p)*laisun(p) + psnsha_wc(p)*laisha(p)
            fpsn_wj(p) = psnsun_wj(p)*laisun(p) + psnsha_wj(p)*laisha(p)
            fpsn_wp(p) = psnsun_wp(p)*laisun(p) + psnsha_wp(p)*laisha(p)
            end if

            if (use_cn) then
            if ( use_c13 ) then
                if (use_c13_timeseries) then
                    rc13_canair(p) = rc13_atm
                else
                    rc13_canair(p) = forc_pc13o2(g)/(forc_pco2(g) - forc_pc13o2(g))
                endif
                rc13_psnsun(p) = rc13_canair(p)/alphapsnsun(p)
                rc13_psnsha(p) = rc13_canair(p)/alphapsnsha(p)
                c13_psnsun(p)  = psnsun(p) * (rc13_psnsun(p)/(1._r8+rc13_psnsun(p)))
                c13_psnsha(p)  = psnsha(p) * (rc13_psnsha(p)/(1._r8+rc13_psnsha(p)))

                ! use fixed c13 ratio with del13C of -25 to test the overall c13 structure
                ! c13_psnsun(p) = 0.01095627 * psnsun(p)
                ! c13_psnsha(p) = 0.01095627 * psnsha(p)
            endif
            if ( use_c14 ) then

                ! determine latitute sector for radiocarbon bomb spike inputs
                if ( grc%latdeg(g) .ge. 30._r8 ) then
                    sector_c14 = 1
                else if ( grc%latdeg(g) .ge. -30._r8 ) then            
                    sector_c14 = 2
                else
                    sector_c14 = 3
                endif

                c14_psnsun(p) = rc14_atm(sector_c14) * psnsun(p)
                c14_psnsha(p) = rc14_atm(sector_c14) * psnsha(p)
            endif
            end if

        end do

    end associate

    end subroutine PhotosynthesisTotal

    !------------------------------------------------------------------------------
    subroutine Fractionation(bounds, fn, filterp, downreg, &
        atm2lnd_inst, canopystate_inst, solarabs_inst, surfalb_inst, photosyns_inst, &
        phase)
    !
    ! !DESCRIPTION:
    ! C13 fractionation during photosynthesis is calculated here after the nitrogen
    ! limitation is taken into account in the CNAllocation module.
    ! 
    ! As of CLM5, nutrient downregulation occurs prior to photosynthesis via leafcn, so we may
    ! ignore the downregulation term in this and assume that the Ci/Ca used in the photosynthesis
    ! calculation is consistent with that in the isotope calculation
    !
    !!USES:
    use clm_varctl     , only : use_hydrstress
    !
    ! !ARGUMENTS:
    type(bounds_type)      , intent(in)    :: bounds
    integer                , intent(in)    :: fn                   ! size of pft filter
    integer                , intent(in)    :: filterp(fn)          ! patch filter
    real(r8)               , intent(in)    :: downreg( bounds%begp: ) ! fractional reduction in GPP due to N limitation (dimensionless)
    type(atm2lnd_type)     , intent(in)    :: atm2lnd_inst
    type(canopystate_type) , intent(in)    :: canopystate_inst
    type(solarabs_type)    , intent(in)    :: solarabs_inst
    type(surfalb_type)     , intent(in)    :: surfalb_inst
    type(photosyns_type)   , intent(in)    :: photosyns_inst
    character(len=*)       , intent(in)    :: phase                ! 'sun' or 'sha'
    !
    ! !LOCAL VARIABLES:
    real(r8) , pointer :: par_z (:,:)   ! needed for backwards compatiblity
    real(r8) , pointer :: alphapsn (:)  ! needed for backwards compatiblity
    real(r8) , pointer :: gs_mol(:,:)   ! leaf stomatal conductance (umol H2O/m**2/s)
    real(r8) , pointer :: an(:,:)       ! net leaf photosynthesis (umol CO2/m**2/s)
    integer  :: f,p,c,g,iv              ! indices
    real(r8) :: co2(bounds%begp:bounds%endp)  ! atmospheric co2 partial pressure (pa)
    real(r8) :: ci
    !------------------------------------------------------------------------------

    call shr_assert_all((ubound(downreg) == (/bounds%endp/)), file= sourcefile, line= 2181)

    associate(                                                  &
            forc_pbot   => atm2lnd_inst%forc_pbot_downscaled_col , & ! Input:  [real(r8) (:)   ]  atmospheric pressure (Pa)
            forc_pco2   => atm2lnd_inst%forc_pco2_grc            , & ! Input:  [real(r8) (:)   ]  partial pressure co2 (Pa)

            c3psn       => pftcon%c3psn                          , & ! Input:  photosynthetic pathway: 0. = c4, 1. = c3

            nrad        => surfalb_inst%nrad_patch               , & ! Input:  [integer  (:)   ]  number of canopy layers, above snow for radiative transfer

            gb_mol      => photosyns_inst%gb_mol_patch             & ! Input:  [real(r8) (:)   ]  leaf boundary layer conductance (umol H2O/m**2/s)
            )

        if (phase == 'sun') then
            par_z    =>    solarabs_inst%parsun_z_patch     ! Input :  [real(r8) (:,:)] par absorbed per unit lai for canopy layer (w/m**2)
            alphapsn =>    photosyns_inst%alphapsnsun_patch ! Output:  [real(r8) (:)]
            if (use_hydrstress) then
            gs_mol => photosyns_inst%gs_mol_sun_patch    ! Input:   [real(r8) (:,:) ] sunlit leaf stomatal conductance (umol H2O/m**2/s)
            an     => photosyns_inst%an_sun_patch        ! Input:  [real(r8) (:,:) ]  net sunlit leaf photosynthesis (umol CO2/m**2/s)
            else
            gs_mol => photosyns_inst%gs_mol_patch        ! Input:   [real(r8) (:,:) ] leaf stomatal conductance (umol H2O/m**2/s)
            an     => photosyns_inst%an_patch            ! Input:  [real(r8) (:,:) ]  net leaf photosynthesis (umol CO2/m**2/s)
            end if
        else if (phase == 'sha') then
            par_z    =>    solarabs_inst%parsha_z_patch     ! Input :  [real(r8) (:,:)] par absorbed per unit lai for canopy layer (w/m**2)
            alphapsn =>    photosyns_inst%alphapsnsha_patch ! Output:  [real(r8) (:)]
            if (use_hydrstress) then
            gs_mol => photosyns_inst%gs_mol_sha_patch    ! Input:   [real(r8) (:,:) ] shaded leaf stomatal conductance (umol H2O/m**2/s)
            an     => photosyns_inst%an_sha_patch        ! Input:  [real(r8) (:,:) ]  net shaded leaf photosynthesis (umol CO2/m**2/s)
            else
            gs_mol => photosyns_inst%gs_mol_patch        ! Input:   [real(r8) (:,:) ] leaf stomatal conductance (umol H2O/m**2/s)
            an     => photosyns_inst%an_patch            ! Input:  [real(r8) (:,:) ]  net leaf photosynthesis (umol CO2/m**2/s)
            end if
        end if

        do f = 1, fn
            p = filterp(f)
            c= patch%column(p)
            g= patch%gridcell(p)

            co2(p) = forc_pco2(g)
            do iv = 1,nrad(p)
            if (par_z(p,iv) <= 0._r8) then           ! night time
                alphapsn(p) = 1._r8
            else                                     ! day time
                ci = co2(p) - (an(p,iv) * &
                    forc_pbot(c) * &
                    (1.4_r8*gs_mol(p,iv)+1.6_r8*gb_mol(p)) / (gb_mol(p)*gs_mol(p,iv)))
                alphapsn(p) = 1._r8 + (((c3psn(patch%itype(p)) * &
                    (4.4_r8 + (22.6_r8*(ci/co2(p))))) + &
                    ((1._r8 - c3psn(patch%itype(p))) * 4.4_r8))/1000._r8)
            end if
            end do
        end do

    end associate

    end subroutine Fractionation

    !-------------------------------------------------------------------------------
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

    !-------------------------------------------------------------------------------
    function ft(tl, ha) result(ans)
    !
    !!DESCRIPTION:
    ! photosynthesis temperature response
    !
    ! !REVISION HISTORY
    ! Jinyun Tang separated it out from Photosynthesis, Feb. 07/2013
    !
    !!USES
    use clm_varcon  , only : rgas, tfrz
    !
    ! !ARGUMENTS:
    real(r8), intent(in) :: tl  ! leaf temperature in photosynthesis temperature function (K)
    real(r8), intent(in) :: ha  ! activation energy in photosynthesis temperature function (J/mol)
    !
    ! !LOCAL VARIABLES:
    real(r8) :: ans
    !-------------------------------------------------------------------------------

    ans = exp( ha / (rgas*1.e-3_r8*(tfrz+25._r8)) * (1._r8 - (tfrz+25._r8)/tl) )

    return
    end function ft

    !-------------------------------------------------------------------------------
    function fth(tl,hd,se,scaleFactor) result(ans)
    !
    !!DESCRIPTION:
    !photosynthesis temperature inhibition
    !
    ! !REVISION HISTORY
    ! Jinyun Tang separated it out from Photosynthesis, Feb. 07/2013
    !
    use clm_varcon  , only : rgas, tfrz
    !
    ! !ARGUMENTS:
    real(r8), intent(in) :: tl  ! leaf temperature in photosynthesis temperature function (K)
    real(r8), intent(in) :: hd  ! deactivation energy in photosynthesis temperature function (J/mol)
    real(r8), intent(in) :: se  ! entropy term in photosynthesis temperature function (J/mol/K)
    real(r8), intent(in) :: scaleFactor  ! scaling factor for high temperature inhibition (25 C = 1.0)
    !
    ! !LOCAL VARIABLES:
    real(r8) :: ans
    !-------------------------------------------------------------------------------

    ans = scaleFactor / ( 1._r8 + exp( (-hd+se*tl) / (rgas*1.e-3_r8*tl) ) )

    return
    end function fth

    !-------------------------------------------------------------------------------
    function fth25(hd,se)result(ans)
    !
    !!DESCRIPTION:
    ! scaling factor for photosynthesis temperature inhibition
    !
    ! !REVISION HISTORY:
    ! Jinyun Tang separated it out from Photosynthesis, Feb. 07/2013
    !
    !!USES
    use clm_varcon  , only : rgas, tfrz
    !
    ! !ARGUMENTS:
    real(r8), intent(in) :: hd    ! deactivation energy in photosynthesis temperature function (J/mol)
    real(r8), intent(in) :: se    ! entropy term in photosynthesis temperature function (J/mol/K)
    !
    ! !LOCAL VARIABLES:
    real(r8) :: ans
    !-------------------------------------------------------------------------------

    ans = 1._r8 + exp( (-hd+se*(tfrz+25._r8)) / (rgas*1.e-3_r8*(tfrz+25._r8)) )

    return
    end function fth25

    !------------------------------------------------------------------------------
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
    
end module PhotosynthesisMod
