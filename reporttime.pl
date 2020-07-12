#!/usr/bin/perl

use v5.10;
use List::Util qw/sum/;
use strict;

chomp(my @data = <>);
@data = sort {$a <=> $b} @data;

my $cut = int 0.5 + (scalar @data) * 0.20;
@data = @data[$cut..$#data - $cut];
say sum(@data)/ scalar(@data);
