create table hosts (
  id integer primary key autoincrement,
  cpuname text not null unique,
  cpuclock text null,
  cpuclockmax text null,
  memclock text null,
  l1dcache text null,
  l1icache text null,
  l2cache text null,
  l3cache text null
);

create table libraries (
  id integer primary key autoincrement,
  name text,
  version text,
  unique(name, version)
);

create table toolchains (
  id integer primary key autoincrement,
  compiler text,
  compilervers text,
  libcvers text,
  unique(compiler, compilervers, libcvers)
);

create table configurations (
  id integer primary key autoincrement,
  funct text not null,
  ftype text not null,
  nelem integer not null,
  nrep integer not null,
  vectlev integer not null,
  lbound real not null,
  ubound real not null,
  ilbound real null,
  iubound real null,
  unique(funct, ftype, nelem, nrep, vectlev, lbound, ubound, ilbound, iubound)
);

create table runs (
  id integer primary key autoincrement,
  time timestamp not null default current_timestamp,
  host integer not null references hosts,
  toolchain integer not null references toolchains
);

create table measurements (
  id integer primary key autoincrement,
  run integer references runs,
  library integer not null references libraries,
  configuration integer not null references configurations,
  evalspersec real not null,
  meanevaltime real not null,
  stddev real,
  maxerr real
);
