#!/usr/bin/perl

use v5.10;
use strict;

my $last = -1;

while (<>) {
    chomp;
    last if /^$/;
    die "Wrong value ($last > $_) at line $.\n" if $last > $_;
    $last = $_;
}
say 'OK';
