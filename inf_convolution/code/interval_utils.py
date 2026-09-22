"""Reference-free Arb and rational serialization helpers."""
from fractions import Fraction
from flint import arb

EXPORT_BITS=100


def ball(q):
    q=Fraction(q)
    return arb(q.numerator)/q.denominator


def interval(lo,hi):return arb(ball((lo+hi)/2),ball((hi-lo)/2))


def ceil_int(x):return int(x.upper().ceil().unique_fmpz())


def floor_int(x):return int(x.lower().floor().unique_fmpz())


def upper_record(x):
    n=ceil_int(x*(1<<EXPORT_BITS))
    return dict(numerator=n,denominator=1<<EXPORT_BITS,decimal=n/(1<<EXPORT_BITS))


def lower_record(x):
    n=floor_int(x*(1<<EXPORT_BITS))
    return dict(numerator=n,denominator=1<<EXPORT_BITS,decimal=n/(1<<EXPORT_BITS))
