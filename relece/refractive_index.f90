! Calculate the refractive index using the Appleton-Hartree equation.
subroutine calc_cold_refractive_index(n, status, frequency, angle, wpe, wce, collision_rate, mode)
    use, intrinsic :: ieee_arithmetic
    implicit none

    integer, parameter :: dp = selected_real_kind(15, 307)
    real(dp), parameter :: PI = 3.141592653589793_dp
    real(dp), parameter :: EPSILON = 1.e-12_dp
    real(dp), intent(in) :: frequency, angle, wpe, wce, collision_rate
    character(1), intent(in) :: mode

    complex(dp), intent(out) :: n
    integer, intent(out) :: status

    real(dp) :: w
    complex(dp) :: wpe2, wce2, delta, n2_numerator, n2_denominator, n2

    !f2py intent(in) frequency
    !f2py intent(in) angle
    !f2py intent(in) wpe
    !f2py intent(in) wce
    !f2py intent(in) collision_rate
    !f2py intent(in) mode
    !f2py intent(out) :: n
    !f2py intent(out) :: status

    w = 2.0_dp * PI * frequency
    wpe2 = wpe**2 * w / cmplx(w, collision_rate, dp)
    wce2 = wce**2 * w / cmplx(w, collision_rate, dp)

    delta = sqrt(wce2 * sin(angle)**4 + 4 * (w**2 - wpe2)**2 * cos(angle)**2 / w**2)
    if (mode == 'X') then
        delta = -delta
    else if (mode /= 'O') then
        status = 1
        return
    end if

    n2_numerator = 2.0_dp * wpe2 * (w**2 - wpe2) / w**2
    n2_denominator = 2.0_dp * (w**2 - wpe2) - wce2 * sin(angle)**2 + sqrt(wce2) * delta
    if (abs(n2_denominator) < EPSILON) then
        ! This singularity marks a resonance condition.
        n = cmplx(ieee_value(0.0_dp, ieee_positive_inf), 0.0_dp, dp)
        return
    endif

    n2 = 1.0_dp - n2_numerator / n2_denominator
    n = sqrt(n2)
end subroutine calc_cold_refractive_index
