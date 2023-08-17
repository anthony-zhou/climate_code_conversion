program parameter_sweep
    ! Note: this parameter sweep doesn't actually work, because sometimes it gets the `no real roots` error. 
    ! For now let's just rely on unit tests.

    use PhotosynthesisMod, only: ci_func

    implicit none
    integer,parameter :: r8 = selected_real_kind(12) ! 8 byte real

    real(r8), dimension(2) :: ci_range, lmr_z_range, par_z_range, gb_mol_range, je_range, rh_can_range
    real(r8) :: ci, lmr_z, par_z, gb_mol, je, rh_can, fval, gs_mol
    real(r8) :: cair, oair
    integer :: p, iv, c, i, j, k, l, m, n
    open(1, file='output_quadratic.txt')

    ! Parameter ranges
    ci_range = [300._r8, 800.0_r8]
    lmr_z_range = [0.1_r8, 10.0_r8]
    par_z_range = [100._r8, 1500._r8]
    gb_mol_range = [100._r8, 1000._r8]
    je_range = [20._r8, 150._r8]
    rh_can_range = [1._r8, 100._r8]

    ! Constants
    p = 1
    iv = 1
    c = 1
    cair = 40._r8
    oair = 21000._r8


    write(1, *) "ci", "lmr_z", "par_z", "gb_mol", "cair", "oair", "je", "rh_can", "fval", "gs_mol"


    ! Parameter sweep
    do i=0,9
        ci = ci_range(1) + (ci_range(2) - ci_range(1)) * i / 9
        do j=0,9
            lmr_z = lmr_z_range(1) + (lmr_z_range(2) - lmr_z_range(1)) * j / 9
            do k=0,9
                par_z = par_z_range(1) + (par_z_range(2) - par_z_range(1)) * k / 9
                do l=0,9
                    gb_mol = gb_mol_range(1) + (gb_mol_range(2) - gb_mol_range(1)) * l / 9
                    do m=0,9
                        je = je_range(1) + (je_range(2) - je_range(1)) * m / 9
                        do n=0,9
                            rh_can = rh_can_range(1) + (rh_can_range(2) - rh_can_range(1)) * n / 9
                            call ci_func(ci, fval, p, iv, c, gb_mol, je, cair, oair, lmr_z, par_z, rh_can, gs_mol)
                            write(1, *) ci, lmr_z, par_z, gb_mol, cair, oair, je, rh_can, fval, gs_mol
                        end do
                    end do
                end do
            end do
        end do
    end do
    close(1)

end program parameter_sweep