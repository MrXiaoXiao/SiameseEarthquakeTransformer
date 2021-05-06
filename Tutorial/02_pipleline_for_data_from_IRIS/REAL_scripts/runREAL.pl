#!/usr/bin/perl -w
$year = YEAR_KEY;
$mon = MON_KEY;
$day = DAY_KEY;

$D = "$year/$mon/$day";
$R = R_KEY;
$G = G_KEY;
$V = V_KEY;
$S = S_KEY;

$dir = DIR_KEY;
$station = STATION_KEY;
$ttime = TTIME_KEY;

system("REAL -D$D -R$R -G$G -S$S -V$V $station $dir $ttime");
print"REAL -D$D -R$R -G$G -S$S -V$V $station $dir $ttime\n";