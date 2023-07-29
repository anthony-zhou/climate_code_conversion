module fib_module
    implicit none
    private
    public :: fibonacci

contains

    integer function sum_two_numbers(a, b)
        integer, intent(in) :: a, b

        sum_two_numbers = a + b
    end function sum_two_numbers

    integer function fibonacci(n)
        integer, intent(in) :: n
        integer :: i
        integer, dimension(:), allocatable :: fib

        allocate(fib(n))

        fib(1) = 0
        fib(2) = 1

        do i = 3, n
            fib(i) = sum_two_numbers(fib(i-1), fib(i-2))
            fib(i) = sum_two_numbers(fib(i-1), fib(i-2))
        end do

        fibonacci = fib(n)

        deallocate(fib)
    end function fibonacci


end module fib_module
