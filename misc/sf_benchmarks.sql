create table hosts (
  id integer primary key autoincrement,
  name text not null unique,
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
  version text
);

create table toolchain (
  id integer primary key autoincrement,
  unamea text
  compiler text
  compilervers text
  libc text
);

create table configurations (
  id integer primary key autoincrement,
  library integer not null references libraries,
  funct text not null,
  ftype text not null,
  nelem integer not null,
  nrep integer not null,
  vectlev integer not null,
  lbound real not null,
  ubound real not null,
  ilbound real null,
  iubound real null
);

create table runs (
  id integer primary key autoincrement,
  time timestamp not null default current_timestamp,
  host integer not null references hosts
);

create table measurements (
  id integer primary key autoincrement,
  run integer references runs,
  test integer not null references tests,
  evalspersec real not null,
  meanevaltime real not null,
  stddev real,
  maxerr real
);
