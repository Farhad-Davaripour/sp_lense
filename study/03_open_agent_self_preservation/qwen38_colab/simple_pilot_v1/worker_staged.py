"""Reuse the reviewed candidate runner with an explicitly staged fit function."""
import study_worker as study
from staged_initiation import fit
import preservation_runner

study.fit = fit

if __name__=='__main__':
    preservation_runner.main()
