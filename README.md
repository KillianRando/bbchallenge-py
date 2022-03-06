# bbchallenge-py

Python tooling to manipulate and visualise the Turing Machines of the bbchallenge project.

## Downloading the database

You can download the database of all 5-state undecided Turing machines here:

- [https://dna.hamilton.ie/tsterin/all_5_states_undecided_machines_with_global_header.zip](https://dna.hamilton.ie/tsterin/all_5_states_undecided_machines_with_global_header.zip)
- [ipfs://QmcgucgLRjAQAjU41w6HR7GJbcte3F14gv9oXcf8uZ8aFM](ipfs://QmcgucgLRjAQAjU41w6HR7GJbcte3F14gv9oXcf8uZ8aFM)
- [https://ipfs.prgm.dev/ipfs/QmcgucgLRjAQAjU41w6HR7GJbcte3F14gv9oXcf8uZ8aFM](https://ipfs.prgm.dev/ipfs/QmcgucgLRjAQAjU41w6HR7GJbcte3F14gv9oXcf8uZ8aFM)

The format of the database is described here: [https://github.com/bbchallenge/bbchallenge-seed](https://github.com/bbchallenge/bbchallenge-seed).

Database shasum: 
  - all_5_states_undecided_machines_with_global_header.zip: `2576b647185063db2aa3dc2f5622908e99f3cd40`
  - all_5_states_undecided_machines_with_global_header: `e57063afefd900fa629cfefb40731fd083d90b5e`
  
## Downloading the index of currently undecided machines

- [https://dna.hamilton.ie/tsterin/bb5_undecided_index](https://dna.hamilton.ie/tsterin/bb5_undecided_index)

Index file shasum: `f9e7f731532259691cf917ff35fd5051c00f1636`

## Getting started

The notebook `BB5.ipynb` is currently self-contained.

You can install all the dependencies (including jupyter lab) by running the following:

```
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
