/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
      Contributing author: Hao Jiang (Princeton U), hjian3@gmail.com
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "pair_lj_43_wall.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "domain.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairLJ43Wall::PairLJ43Wall(LAMMPS *lmp) : Pair(lmp)
{
  respa_enable = 0;
}

/* ---------------------------------------------------------------------- */

PairLJ43Wall::~PairLJ43Wall()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);

    memory->destroy(cutz);

    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(foffset);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairLJ43Wall::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r2inv,r3inv,r4inv,forcelj,factor_lj;
  double r,t;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double * xi, *xj;
  double xc[3];

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  tagint *tag = atom->tag;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

 // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xi = x[i]; 

    xc[0] = x[i][0];
    xc[1] = x[i][1];
    xc[2] = x[i][2];

    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
   
    domain->remap(xc);
    for (jj = 0; jj < jnum; jj++) {
  
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;
      xj = x[j];

      if(xj[0] > domain->boxhi[0] || xj[0] < domain->boxlo[0]) continue;
      if(xj[1] > domain->boxhi[1] || xj[1] < domain->boxlo[1]) continue;
      if(xj[2] > domain->boxhi[2] || xj[2] < domain->boxlo[2]) continue;

      delx = 0;
      dely = 0;
 
      delz = xc[2] - xj[2];
       
      rsq = delz*delz; 
          
      jtype = type[j];

      r = sqrt(rsq);
        
       //  if (rsq < cutsq[itype][jtype]) {
        
       if(r < cutz[itype][jtype]) {
 
  
        r2inv = 1.0/rsq;
        r4inv = r2inv*r2inv;
        r3inv = sqrt(r4inv*r2inv);
        t = r/cutz[itype][jtype];

        forcelj = lj1[itype][jtype]*r4inv - lj2[itype][jtype]*r3inv - 
          t*foffset[itype][jtype];
        fpair = factor_lj*forcelj*r2inv;

        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
         f[j][2] -= delz*fpair;     
       }

        if (eflag) {
          evdwl = lj3[itype][jtype]*r4inv-lj4[itype][jtype]*r3inv +
            (t-1.0)*foffset[itype][jtype] - offset[itype][jtype];
          evdwl *= factor_lj;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
        }
      //}
    } 

  }
      if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJ43Wall::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");

  memory->create(cutz,n+1,n+1,"pair:cutzdirection");

  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(foffset,n+1,n+1,"pair:foffset");
  memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJ43Wall::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = force->numeric(FLERR,arg[0]);

  if (cut_global <= 0.0)
    error->all(FLERR,"Illegal pair_style command");

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJ43Wall::coeff(int narg, char **arg)
{
  if (narg < 5 || narg > 6)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(arg[0],atom->ntypes,ilo,ihi);
  force->bounds(arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);

  double cut_one = cut_global;
  double zcut;
  cut_one = force->numeric(FLERR,arg[4]);
  zcut    = force->numeric(FLERR,arg[5]);

  if (cut_one <= 0.0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      cutz[i][j] = zcut;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJ43Wall::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
    cutz[i][j] = mix_distance(cutz[i][i],cutz[j][j]);
  }

  lj1[i][j] = 4.0 * epsilon[i][j] ;
  lj2[i][j] = 3.0 * sigma[i][j];
  lj3[i][j] = epsilon[i][j];
  lj4[i][j] = sigma[i][j];

  
  foffset[i][j] = lj1[i][j]/pow(cutz[i][j],4.0) - lj2[i][j]/pow(cutz[i][j],3.0);

  offset[i][j] = lj3[i][j]/pow(cutz[i][j],4.0) - lj4[i][j]/pow(cutz[i][j],3.0);

//  offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,9.0) - pow(ratio,6.0));

  printf("%d %d %f %f %f %f\n", i, j, lj1[i][j], lj2[i][j], lj3[i][j], lj4[i][j]);
  printf("%d %d %f %f %f\n", i, j, foffset[i][j], offset[i][j], cutz[i][j]);

  cut[j][i] = cut[i][j];
  cutz[j][i] = cutz[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  foffset[j][i] = foffset[i][j];
  offset[j][i] = offset[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJ43Wall::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
        fwrite(&cutz[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJ43Wall::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&epsilon[i][j],sizeof(double),1,fp);
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
          fread(&cutz[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cutz[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJ43Wall::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJ43Wall::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

double PairLJ43Wall::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj,
                         double &fforce)
{
  double r2inv,r3inv,r4inv,forcelj,philj,r,t;

  r2inv = 1.0/rsq;
  r4inv = r2inv*r2inv;
  r3inv = sqrt(r2inv*r4inv);
  r = sqrt(rsq);
  t = r/cutz[itype][jtype];
 
  forcelj = lj1[itype][jtype]*r4inv - lj2[itype][jtype]*r3inv -
          t*foffset[itype][jtype];

  fforce = factor_lj*forcelj*r2inv;

  philj = r4inv*lj3[itype][jtype] -lj4[itype][jtype]*r3inv +
    (t-1.0)*foffset[itype][jtype] - offset[itype][jtype];

  return factor_lj*philj;
}
