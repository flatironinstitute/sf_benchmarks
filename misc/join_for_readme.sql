.mode html
.headers on

SELECT
    configurations.func,
    libraries.name,
    configurations.ftype,
    measurements.nelem,
    measurements.veclev,
    configurations.lbound,
    configurations.ubound,
    measurements.megaevalspersec
FROM
    configurations
JOIN measurements
ON   configurations.id==measurements.configuration
JOIN libraries
ON   libraries.id==measurements.library
JOIN runs
ON   runs.id==measurements.run
WHERE
     (measurements.nelem==1024 OR measurements.nrepeat==1) AND
     runs.id==(SELECT MAX(rowid) FROM runs)
ORDER BY configurations.func, configurations.ftype, measurements.nelem, measurements.megaevalspersec DESC;
