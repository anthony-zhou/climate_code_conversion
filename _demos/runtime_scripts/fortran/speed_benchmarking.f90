program speed_benchmarking
    use PhotosynthesisMod, only: hybrid

    implicit none
    double precision :: start_time, end_time, elapsed_time
    integer :: i, n, j
    integer,parameter :: r8 = selected_real_kind(12) ! 8 byte real


    real(r8)     :: ci       ! intracellular leaf CO2 (Pa)
    real(r8)     :: lmr_z    ! canopy layer: leaf maintenance respiration rate (umol CO2/m**2/s)
    real(r8)     :: par_z    ! par absorbed per unit lai for canopy layer (w/m**2)
    real(r8)     :: gb_mol   ! leaf boundary layer conductance (umol H2O/m**2/s)
    real(r8)     :: je       ! electron transport rate (umol electrons/m**2/s)
    real(r8)     :: cair     ! Atmospheric CO2 partial pressure (Pa)
    real(r8)     :: oair     ! Atmospheric O2 partial pressure (Pa)
    real(r8)     :: rh_can   ! canopy air realtive humidity
    integer      :: p, iv, c ! pft, vegetation type and column indexes (UNUSED)

    real(r8)     :: fval     ! return function of the value f(ci)
    real(r8)     :: gs_mol   ! leaf stomatal conductance (umol H2O/m**2/s)
    integer :: iter              !number of iterations used, for record only

    integer :: grid_sizes(6) ! grid sizes to for runtimes


    integer :: unit_number
    character(len=40) :: filename

    filename = 'fortran_runtime.txt'
    unit_number = 10
    
    open(unit=unit_number, file=filename, action='write')    


    ci = 40
    lmr_z = 4
    par_z = 500
    gb_mol = 50000
    je = 40
    cair = 45
    oair = 21000
    rh_can = 0.40
    p = 1
    iv = 1
    c = 1
    

    grid_sizes = (/1000, 10000, 100000, 1000000, 10000000, 100000000/)
    

    do i = 1, size(grid_sizes)
        n = grid_sizes(i)
        call cpu_time(start_time)
        do j = 0, n
            call hybrid(ci, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z, rh_can, gs_mol, iter)
        end do
        call cpu_time(end_time)
        elapsed_time = end_time - start_time
        write(unit_number, *) ci, ',', gs_mol, ',', n, ',', elapsed_time
    end do
    
    close(unit_number)

    ! print*, 'Elapsed CPU time = ', elapsed_time, ' seconds'

end program speed_benchmarking