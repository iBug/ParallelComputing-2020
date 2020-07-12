#!/usr/bin/perl

use List::Util qw/sum/;
use strict;

my @data = sort {$a <=> $b} <>;
die "Empty content?\n" unless @data;
my $cut = int 0.5 + scalar @data * 0.20;
@data = @data[$cut..$#data - $cut];
printf "%.6f\n", sum(@data)/ scalar @data;
