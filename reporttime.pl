#!/usr/bin/perl

use v5.10;
use List::Util qw/sum/;
use strict;

my @data = sort {$a <=> $b} <>;
die "Empty content?\n" unless @data;
my $cut = int 0.5 + scalar @data * 0.20;
@data = @data[$cut..$#data - $cut];
say sum(@data)/ scalar @data;
