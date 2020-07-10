#!/usr/bin/perl

use v5.10;
use strict;

my $last = -1;

while (<>) {
    chomp;
    die "Wrong value ($last > $_) at line $.\n" if $last > $_;
    $last = $_;
}
