      subroutine fort_bessel_jn(n, x, y)
      INTEGER*4 n
      REAL*8 x,y
      y = BESSEL_JN(n, x)
      end subroutine

      subroutine fort_bessel_yn(n, x, y)
      INTEGER*4 n
      REAL*8 x,y
      y = BESSEL_YN(n, x)
      end subroutine
