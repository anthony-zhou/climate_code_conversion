module fac
  implicit none
  
  contains

  recursive function factorial(n) result(fact)
    integer, intent(in) :: n
    integer :: fact

    if (n == 0) then
      fact = 1
    else
      fact = n * factorial(n - 1)
    end if

  end function factorial

end module fac
