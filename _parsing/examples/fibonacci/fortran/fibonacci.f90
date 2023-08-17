module fib_module
    implicit none
    private
    public :: fibonacci

contains

    function sum_two_numbers(a, b) result(sum)
        integer, intent(in) :: a, b
        integer :: sum

        sum = a + b
    end function sum_two_numbers

    function fibonacci(n) result(fib_value)
        integer, intent(in) :: n
        integer :: fib_value
        integer :: i
        integer, dimension(:), allocatable :: fib

        allocate(fib(n))

        fib(1) = 0
        fib(2) = 1

        do i = 3, n
            fib(i) = sum_two_numbers(fib(i-1), fib(i-2))
        end do

        fib_value = fib(n)

        deallocate(fib)
    end function fibonacci


end module fib_module
