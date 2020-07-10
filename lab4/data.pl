#!/usr/bin/perl

use v5.10;
use strict;

die "Need an argument\n" unless @ARGV > 0;

my $n = int $ARGV[0];
my $max;

die "Need a positive integer\n" unless $n > 0;

if (@ARGV > 1) {
    $max = int $ARGV[1];
} else {
    $max = 2 * $n;
}

sub random {
    return int rand $_[0];
}

say $n;
say random $max for 1 .. $n;
