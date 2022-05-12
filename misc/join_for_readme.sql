.mode html
.headers on

SELECT
    configurations.func,
    libraries.name,
    configurations.ftype,
    measurements.nelem,
    measurements.veclev,
    ROUND(configurations.lbound, 2),
    ROUND(configurations.ubound, 2),
    ROUND(measurements.megaevalspersec, 1),
    ROUND(measurements.cyclespereval, 1)
FROM
    configurations
JOIN measurements
ON   configurations.id=measurements.configuration
JOIN libraries
ON   libraries.id=measurements.library
WHERE
     (measurements.nelem=1024 OR measurements.nrepeat=1) AND
     measurements.run=(SELECT MIN(id) FROM runs)
ORDER BY configurations.func, configurations.ftype, measurements.nelem, measurements.megaevalspersec DESC;
