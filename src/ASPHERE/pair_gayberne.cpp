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
   Contributing author: Mike Brown (SNL)
------------------------------------------------------------------------- */

#include "pair_gayberne.h"
#include <mpi.h>
#include <cmath>
#include "math_extra.h"
#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "citeme.h"
#include "memory.h"
#include "error.h"
#include "utils.h"

/* ----------------------------------------------------------------------
   Additional #include
------------------------------------------------------------------------- */
#include <cstring>
#include "fix.h"
#include "fix_neigh_history.h"
#include "modify.h"
#include "update.h" 
/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

static const char cite_pair_gayberne[] =
  "pair gayberne command:\n\n"
  "@Article{Brown09,\n"
  " author = {W. M. Brown, M. K. Petersen, S. J. Plimpton, and G. S. Grest},\n"
  " title =  {Liquid crystal nanodroplets in solution},\n"
  " journal ={J.~Chem.~Phys.},\n"
  " year =   2009,\n"
  " volume = 130,\n"
  " pages =  {044901}\n"
  "}\n\n";

/* ---------------------------------------------------------------------- */

PairGayBerne::PairGayBerne(LAMMPS *lmp) : Pair(lmp)
{
  if (lmp->citeme) lmp->citeme->add(cite_pair_gayberne);

  single_enable = 0;
  writedata = 1;

/* ----------------------------------------------------------------------
   For calculation of friction
------------------------------------------------------------------------- */
  size_history = 3; 
  fix_history = NULL; 
  use_history = 1; 
  beyond_contact = 1; 
  no_virial_fdotr_compute = 1; 
  nondefault_history_transfer = 0;
/* ---------------------------------------------------------------------- */
}

/* ----------------------------------------------------------------------
   free all arrays
------------------------------------------------------------------------- */

PairGayBerne::~PairGayBerne()
{

/* ----------------------------------------------------------------------
   For calculation of friction
------------------------------------------------------------------------- */
   if (fix_history) modify->delete_fix("NEIGH_HISTORY"); 
/* ---------------------------------------------------------------------- */

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(form);
    memory->destroy(gammasb);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(shape1);
    memory->destroy(shape2);
    memory->destroy(well);
    memory->destroy(cut);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);

/* ----------------------------------------------------------------------
   New variables
------------------------------------------------------------------------- */
    memory->destroy(cfa);
    memory->destroy(cfb);
    memory->destroy(cfc);
    memory->destroy(hth);
    memory->destroy(aq);
    memory->destroy(bq);
    memory->destroy(cq);
    memory->destroy(kn);
    memory->destroy(kt);
    memory->destroy(gamma_n); 
    memory->destroy(gamma_t);
    memory->destroy(xmu); 
/* ---------------------------------------------------------------------- */

    delete [] lshape;
    delete [] setwell;
  }
}

/* ---------------------------------------------------------------------- */

void PairGayBerne::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double evdwl,one_eng,rsq,r2inv,r6inv,forcelj,factor_lj;
  double fforce[3],ttor[3],rtor[3],r12[3];
  double a1[3][3],b1[3][3],g1[3][3],a2[3][3],b2[3][3],g2[3][3],temp[3][3];
  int *ilist,*jlist,*numneigh,**firstneigh;
  double *iquat,*jquat;

/* ----------------------------------------------------------------------
   New variables
------------------------------------------------------------------------- */
  int k;
  double nveci[3], nvecj[3], dot_ni_r12, dot_nj_r12, norm_ni, norm_nj, 
  norm_r12, cosi, cosj;
  double iweight, jweight;
  int *touch,**firsttouch;
  double *history,*allhistory,**firsthistory; 
/* ---------------------------------------------------------------------- */

  evdwl = 0.0;
  ev_init(eflag,vflag);

  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  double **x = atom->x;
  double **f = atom->f;
  double **tor = atom->torque;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

/* ----------------------------------------------------------------------
   For calculation of friction
------------------------------------------------------------------------- */
  firsttouch = fix_history->firstflag;
  firsthistory = fix_history->firstvalue;
  ntimestep = update->ntimestep; 
/* ---------------------------------------------------------------------- */

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];

    //if (form[itype][itype] == ELLIPSE_ELLIPSE) {
      iquat = bonus[ellipsoid[i]].quat;
      MathExtra::quat_to_mat_trans(iquat,a1);
      /*MathExtra::diag_times3(well[itype],a1,temp);
      MathExtra::transpose_times3(a1,temp,b1);
      MathExtra::diag_times3(shape2[itype],a1,temp);
      MathExtra::transpose_times3(a1,temp,g1);*/
    //}

/* ----------------------------------------------------------------------
   For calculation of friction
------------------------------------------------------------------------- */
    touch = firsttouch[i];
    allhistory = firsthistory[i]; 
/* ---------------------------------------------------------------------- */

    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      // r12 = center to center vector

      r12[0] = x[j][0]-x[i][0];
      r12[1] = x[j][1]-x[i][1];
      r12[2] = x[j][2]-x[i][2];
      rsq = MathExtra::dot3(r12,r12);
      jtype = type[j];

      // compute if less than cutoff

      if (rsq < cutsq[itype][jtype]) {

        switch (form[itype][jtype]) {
        /*case SPHERE_SPHERE:
          r2inv = 1.0/rsq;
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
          forcelj *= -r2inv;
          if (eflag) one_eng =
                       r6inv*(r6inv*lj3[itype][jtype]-lj4[itype][jtype]) -
                       offset[itype][jtype];
          fforce[0] = r12[0]*forcelj;
          fforce[1] = r12[1]*forcelj;
          fforce[2] = r12[2]*forcelj;
          ttor[0] = ttor[1] = ttor[2] = 0.0;
          rtor[0] = rtor[1] = rtor[2] = 0.0;
          break;

        case SPHERE_ELLIPSE:
          jquat = bonus[ellipsoid[j]].quat;
          MathExtra::quat_to_mat_trans(jquat,a2);
          MathExtra::diag_times3(well[jtype],a2,temp);
          MathExtra::transpose_times3(a2,temp,b2);
          MathExtra::diag_times3(shape2[jtype],a2,temp);
          MathExtra::transpose_times3(a2,temp,g2);
          one_eng = gayberne_lj(j,i,a2,b2,g2,r12,rsq,fforce,rtor);
          ttor[0] = ttor[1] = ttor[2] = 0.0;
          break;

        case ELLIPSE_SPHERE:
          one_eng = gayberne_lj(i,j,a1,b1,g1,r12,rsq,fforce,ttor);
          rtor[0] = rtor[1] = rtor[2] = 0.0;
          break;
          */

        default:
          jquat = bonus[ellipsoid[j]].quat;
          MathExtra::quat_to_mat_trans(jquat,a2);
          /*MathExtra::diag_times3(well[jtype],a2,temp);
          MathExtra::transpose_times3(a2,temp,b2);
          MathExtra::diag_times3(shape2[jtype],a2,temp);
          MathExtra::transpose_times3(a2,temp,g2);
          one_eng = gayberne_analytic(i,j,a1,a2,b1,b2,g1,g2,r12,rsq,
                                      fforce,ttor,rtor);*/

/* ----------------------------------------------------------------------
   This part is made to achieve modelling of anisotropic surfa-
   ce charge of kaolinite particles. type=1 represents Silica 
   face, type=2 represents Alumina face, and type=3 represents 
   edge. Important thing is that this code works for only mono-
   disperse sample - it doesn't work for poly disperse sample.
------------------------------------------------------------------------- */
          for(k=0; k<3; k++) nveci[k] = a1[2][k];
          for(k=0; k<3; k++) nvecj[k] = a2[2][k];
          dot_ni_r12 = MathExtra::dot3(nveci,r12);
          dot_nj_r12 = MathExtra::dot3(nvecj,r12);
          norm_ni = sqrt( MathExtra::dot3(nveci,nveci) );
          norm_nj = sqrt( MathExtra::dot3(nvecj,nvecj) );
          norm_r12 = sqrt( rsq );
          if ( norm_ni*norm_r12 != 0.0 ) cosi=dot_ni_r12/(norm_ni*norm_r12);
          else cosi = 0.0;
          iweight = pow( pow(cosi, 2.0), A_const);
          if ( norm_nj*norm_r12 != 0.0 ) cosj=dot_nj_r12/(norm_nj*norm_r12);
          else cosj = 0.0;
          jweight = pow( pow(cosj, 2.0), A_const); 

          /* change atom-type */
          //particle 1
          if( type[i] == 1 || type[i] == 2){
            if( cosi <= 1.0 && cosi >= 0.0 ) type[i]=1;   //Silica face
            else if ( cosi >= -1.0 && cosi < 0.0 ) type[i]=2;//Alumina face
          } 
          /* shape */
          if(shape2[type[i]][0]==1.0 && shape2[type[i]][1]==1.0 
             && shape2[type[i]][2]==1.0){
            for(k=0; k<3; k++) shape1[type[i]][k] = shape1[type[j]][k];
            for(k=0; k<3; k++) shape2[type[i]][k] = shape2[type[j]][k];
            lshape[type[i]] = (shape1[type[i]][0]*shape1[type[i]][1]
                            +shape1[type[i]][2]*shape1[type[i]][2]) 
                            *sqrt(shape1[type[i]][0]*shape1[type[i]][1]);
          } 
          //particle 2
          if( type[j] == 1 || type[j] == 2 ){
            if( cosj <= 1.0 && cosj >= 0.0 ) type[j]=2;   //Alumina face
            else if ( cosj >= -1.0 && cosj < 0.0 ) type[j]=1;//Silica face
          } 
          /* shape */
          if(shape2[type[j]][0]==1.0 && shape2[type[j]][1]==1.0 
             && shape2[type[j]][2]==1.0){
            for(k=0; k<3; k++) shape1[type[j]][k] = shape1[type[i]][k];
            for(k=0; k<3; k++) shape2[type[j]][k] = shape2[type[i]][k];
            lshape[type[j]] = (shape1[type[j]][0]*shape1[type[j]][1]
                            +shape1[type[j]][2]*shape1[type[j]][2]) 
                            *sqrt(shape1[type[j]][0]*shape1[type[j]][1]);
          }

          MathExtra::diag_times3(shape2[itype],a1,temp);
          MathExtra::transpose_times3(a1,temp,g1);
          MathExtra::diag_times3(shape2[jtype],a2,temp);
          MathExtra::transpose_times3(a2,temp,g2);
          
          MathExtra::diag_times3(well[type[i]],a1,temp);
          MathExtra::transpose_times3(a1,temp,b1);
          MathExtra::diag_times3(well[type[j]],a2,temp);
          MathExtra::transpose_times3(a2,temp,b2); 
/* ---------------------------------------------------------------------- */

          one_eng = gayberne_analytic(i,j,a1,a2,b1,b2,g1,g2,r12,rsq,fforce,
                    ttor,rtor, iweight,jweight,nveci,nvecj,cosi,cosj, touch, 
                    history, allhistory, jj);
          break;
        }

        fforce[0] *= factor_lj;
        fforce[1] *= factor_lj;
        fforce[2] *= factor_lj;
        ttor[0] *= factor_lj;
        ttor[1] *= factor_lj;
        ttor[2] *= factor_lj;

        f[i][0] += fforce[0];
        f[i][1] += fforce[1];
        f[i][2] += fforce[2];
        tor[i][0] += ttor[0];
        tor[i][1] += ttor[1];
        tor[i][2] += ttor[2];

        if (newton_pair || j < nlocal) {
          rtor[0] *= factor_lj;
          rtor[1] *= factor_lj;
          rtor[2] *= factor_lj;
          f[j][0] -= fforce[0];
          f[j][1] -= fforce[1];
          f[j][2] -= fforce[2];
          tor[j][0] += rtor[0];
          tor[j][1] += rtor[1];
          tor[j][2] += rtor[2];
        }

        if (eflag) evdwl = factor_lj*one_eng;

        if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,
                                 evdwl,0.0,fforce[0],fforce[1],fforce[2],
                                 -r12[0],-r12[1],-r12[2]);
/* ----------------------------------------------------------------------
   For calculation of friction
------------------------------------------------------------------------- */
      }else{
        touch[jj] = 0;
        history = &allhistory[size_history*jj];
        for (int k = 0; k < size_history; k++) history[k] = 0.0; 
/* ---------------------------------------------------------------------- */
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairGayBerne::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(form,n+1,n+1,"pair:form");
  memory->create(gammasb,n+1,n+1,"pair:gammasb");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(shape1,n+1,3,"pair:shape1");
  memory->create(shape2,n+1,3,"pair:shape2");
  memory->create(well,n+1,3,"pair:well");
  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");

/* ----------------------------------------------------------------------
   For new variables
------------------------------------------------------------------------- */
  memory->create(cfa,n+1,n+1,"pair:cfa");
  memory->create(cfb,n+1,n+1,"pair:cfb");
  memory->create(cfc,n+1,n+1,"pair:cfc");
  memory->create(hth,n+1,n+1,"pair:hth");
  memory->create(aq,n+1,n+1,"pair:aq");
  memory->create(bq,n+1,n+1,"pair:bq");
  memory->create(cq,n+1,n+1,"pair:cq");
  memory->create(kn,n+1,n+1,"pair:kn");
  memory->create(kt,n+1,n+1,"pair:kt");
  memory->create(gamma_n,n+1,n+1,"pair:gamma_n"); 
  memory->create(gamma_t,n+1,n+1,"pair:gamma_t");
  memory->create(xmu,n+1,n+1,"pair:xmu");
  memory->create(hf,n+1,n+1,"pair:hf"); 
/* ---------------------------------------------------------------------- */

  lshape = new double[n+1];
  setwell = new int[n+1];
  for (int i = 1; i <= n; i++) setwell[i] = 0;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairGayBerne::settings(int narg, char **arg)
{
  /*if (narg != 4) error->all(FLERR,"Illegal pair_style command");*/
  if (narg != 8) error->all(FLERR,"Illegal pair_style command");

  gamma = force->numeric(FLERR,arg[0]);
  upsilon = force->numeric(FLERR,arg[1])/2.0;
  mu = force->numeric(FLERR,arg[2]);
  cut_global = force->numeric(FLERR,arg[3]);

/* ----------------------------------------------------------------------
   For new flags
------------------------------------------------------------------------- */
  Q_flag = force->numeric(FLERR,arg[4]);
  C_flag = force->numeric(FLERR,arg[5]);
  interval = force->numeric(FLERR,arg[6]);
  A_const = force->numeric(FLERR,arg[7]);
/* ---------------------------------------------------------------------- */

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairGayBerne::coeff(int narg, char **arg)
{
  /*if (narg < 10 || narg > 11)*/
  if (narg < 24 || narg > 25)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  /*double epsilon_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);
  double eia_one = force->numeric(FLERR,arg[4]);
  double eib_one = force->numeric(FLERR,arg[5]);
  double eic_one = force->numeric(FLERR,arg[6]);
  double eja_one = force->numeric(FLERR,arg[7]);
  double ejb_one = force->numeric(FLERR,arg[8]);
  double ejc_one = force->numeric(FLERR,arg[9]);

  double cut_one = cut_global;
  if (narg == 11) cut_one = force->numeric(FLERR,arg[10]);*/

/* ----------------------------------------------------------------------
   For new variables
------------------------------------------------------------------------- */
  double gammasb_one = force->numeric(FLERR,arg[2]); 
  double epsilon_one = force->numeric(FLERR,arg[3]);
  double sigma_one = force->numeric(FLERR,arg[4]);
  double eia_one = force->numeric(FLERR,arg[5]);
  double eib_one = force->numeric(FLERR,arg[6]);
  double eic_one = force->numeric(FLERR,arg[7]);
  double eja_one = force->numeric(FLERR,arg[8]);
  double ejb_one = force->numeric(FLERR,arg[9]);
  double ejc_one = force->numeric(FLERR,arg[10]);
  double cfa_one = force->numeric(FLERR,arg[11]);
  double cfb_one = force->numeric(FLERR,arg[12]);
  double cfc_one = force->numeric(FLERR,arg[13]);
  double hth_one = force->numeric(FLERR,arg[14]);
  double aq_one = force->numeric(FLERR,arg[15]);
  double bq_one = force->numeric(FLERR,arg[16]);
  double cq_one = force->numeric(FLERR,arg[17]);
  double kn_one = force->numeric(FLERR,arg[18]);
  double kt_one = force->numeric(FLERR,arg[19]);
  double gamma_n_one = force->numeric(FLERR,arg[20]);
  double gamma_t_one = force->numeric(FLERR,arg[21]);
  double xmu_one = force->numeric(FLERR,arg[22]);
  double hf_one = force->numeric(FLERR,arg[23]);
  double cut_one = cut_global;
  if (narg == 25) cut_one = force->numeric(FLERR,arg[24]); 
/* ---------------------------------------------------------------------- */

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
/* ----------------------------------------------------------------------
   For new variables
------------------------------------------------------------------------- */
      gammasb[i][j] = gammasb_one;
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cfa[i][j] = cfa_one;
      cfb[i][j] = cfb_one;
      cfc[i][j] = cfc_one;
      hth[i][j] = hth_one;
      aq[i][j] = aq_one;
      bq[i][j] = bq_one;
      cq[i][j] = cq_one;
      kn[i][j] = kn_one;
      kt[i][j] = kt_one;
      gamma_n[i][j] = gamma_n_one;
      gamma_t[i][j] = gamma_t_one;
      xmu[i][j] = xmu_one;
      hf[i][j] = hf_one; 
/* ---------------------------------------------------------------------- */

      cut[i][j] = cut_one;
      if (eia_one != 0.0 || eib_one != 0.0 || eic_one != 0.0) {
        well[i][0] = pow(eia_one,-1.0/mu);
        well[i][1] = pow(eib_one,-1.0/mu);
        well[i][2] = pow(eic_one,-1.0/mu);
        if (eia_one == eib_one && eib_one == eic_one) setwell[i] = 2;
        else setwell[i] = 1;
      }
      if (eja_one != 0.0 || ejb_one != 0.0 || ejc_one != 0.0) {
        well[j][0] = pow(eja_one,-1.0/mu);
        well[j][1] = pow(ejb_one,-1.0/mu);
        well[j][2] = pow(ejc_one,-1.0/mu);
        if (eja_one == ejb_one && ejb_one == ejc_one) setwell[j] = 2;
        else setwell[j] = 1;
      }
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}


/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairGayBerne::init_style()
{
  avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  if (!avec) error->all(FLERR,
             "Pair gayberne requires atom style ellipsoid");

  neighbor->request(this,instance_me);

  // per-type shape precalculations
  // require that atom shapes are identical within each type
  // if shape = 0 for point particle, set shape = 1 as required by Gay-Berne

  for (int i = 1; i <= atom->ntypes; i++) {
    if (!atom->shape_consistency(i,shape1[i][0],shape1[i][1],shape1[i][2]))
      error->all(FLERR,
      "Pair gayberne requires atoms with same type have same shape");
    if (shape1[i][0] == 0.0)
      shape1[i][0] = shape1[i][1] = shape1[i][2] = 1.0;
    shape2[i][0] = shape1[i][0]*shape1[i][0];
    shape2[i][1] = shape1[i][1]*shape1[i][1];
    shape2[i][2] = shape1[i][2]*shape1[i][2];
    lshape[i] = (shape1[i][0]*shape1[i][1]+shape1[i][2]*shape1[i][2]) *
      sqrt(shape1[i][0]*shape1[i][1]);
  }

/* ----------------------------------------------------------------------
   For calculation of friction
------------------------------------------------------------------------- */
  dt = update->dt;

  if (use_history && fix_history == NULL) {
    char dnumstr[16];
    sprintf(dnumstr,"%d",size_history);
    char **fixarg = new char*[4];
    fixarg[0] = (char *) "NEIGH_HISTORY";
    fixarg[1] = (char *) "all";
    fixarg[2] = (char *) "NEIGH_HISTORY";
    fixarg[3] = dnumstr;
    modify->add_fix(4,fixarg,1);
    delete [] fixarg;
    fix_history = (FixNeighHistory *) modify->fix[modify->nfix-1];
    fix_history->pair = this;
  }
  // set fix which stores history info
  if (use_history) {
    int ifix = modify->find_fix("NEIGH_HISTORY");
    if (ifix < 0) error->all(FLERR,
                  "Could not find pair fix neigh history ID");
    fix_history = (FixNeighHistory *) modify->fix[ifix];
  } 
/* ---------------------------------------------------------------------- */

}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairGayBerne::init_one(int i, int j)
{
  if (setwell[i] == 0 || setwell[j] == 0)
    error->all(FLERR,"Pair gayberne epsilon a,b,c coeffs are not all set");

  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  if (offset_flag && (cut[i][j] > 0.0)) {
    double ratio = sigma[i][j] / cut[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0)-pow(ratio,6.0));
  } else offset[i][j] = 0.0;

  int ishape = 0;
  if (shape1[i][0] != shape1[i][1] ||
      shape1[i][0] != shape1[i][2] ||
      shape1[i][1] != shape1[i][2]) ishape = 1;
  if (setwell[i] == 1) ishape = 1;
  int jshape = 0;
  if (shape1[j][0] != shape1[j][1] ||
      shape1[j][0] != shape1[j][2] ||
      shape1[j][1] != shape1[j][2]) jshape = 1;
  if (setwell[j] == 1) jshape = 1;

  if (ishape == 0 && jshape == 0)
    form[i][i] = form[j][j] = form[i][j] = form[j][i] = SPHERE_SPHERE;
  else if (ishape == 0) {
    form[i][i] = SPHERE_SPHERE; form[j][j] = ELLIPSE_ELLIPSE;
    form[i][j] = SPHERE_ELLIPSE; form[j][i] = ELLIPSE_SPHERE;
  } else if (jshape == 0) {
    form[j][j] = SPHERE_SPHERE; form[i][i] = ELLIPSE_ELLIPSE;
    form[j][i] = SPHERE_ELLIPSE; form[i][j] = ELLIPSE_SPHERE;
  } else
    form[i][i] = form[j][j] = form[i][j] = form[j][i] = ELLIPSE_ELLIPSE;

/* ----------------------------------------------------------------------
   For new variables
------------------------------------------------------------------------- */
  gammasb[j][i] = gammasb[i][j];
  cfa[j][i] = cfa[i][j]; 
  cfb[j][i] = cfb[i][j];
  cfc[j][i] = cfc[i][j]; 
  hth[j][i] = hth[i][j];
  aq[j][i] = aq[i][j]; 
  bq[j][i] = bq[i][j];
  cq[j][i] = cq[i][j]; 
  kn[j][i] = kn[i][j];
  kt[j][i] = kt[i][j];
  gamma_n[j][i] = gamma_n[i][j];
  gamma_t[j][i] = gamma_t[i][j];
  xmu[j][i] = xmu[i][j];
  hf[j][i] = hf[i][j]; 
/* ---------------------------------------------------------------------- */

  epsilon[j][i] = epsilon[i][j];
  sigma[j][i] = sigma[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  return cut[i][j];

}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairGayBerne::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++) {
    fwrite(&setwell[i],sizeof(int),1,fp);
    if (setwell[i]) fwrite(&well[i][0],sizeof(double),3,fp);
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {

/* ----------------------------------------------------------------------
   For new variables
------------------------------------------------------------------------- */
        fwrite(&gammasb[i][j],sizeof(double),1,fp); 
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cfa[i][j],sizeof(double),1,fp); 
        fwrite(&cfb[i][j],sizeof(double),1,fp);
        fwrite(&cfc[i][j],sizeof(double),1,fp);
        fwrite(&hth[i][j],sizeof(double),1,fp);
        fwrite(&aq[i][j],sizeof(double),1,fp); 
        fwrite(&bq[i][j],sizeof(double),1,fp);
        fwrite(&cq[i][j],sizeof(double),1,fp);
        fwrite(&kn[i][j],sizeof(double),1,fp);
        fwrite(&kt[i][j],sizeof(double),1,fp);
        fwrite(&gamma_n[i][j],sizeof(double),1,fp);
        fwrite(&gamma_t[i][j],sizeof(double),1,fp);
        fwrite(&xmu[i][j],sizeof(double),1,fp);
        fwrite(&hf[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp); 
/* ---------------------------------------------------------------------- */

      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairGayBerne::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++) {
    if (me == 0) utils::sfread(FLERR,&setwell[i],sizeof(int),1,
                               fp,NULL,error);
    MPI_Bcast(&setwell[i],1,MPI_INT,0,world);
    if (setwell[i]) {
      if (me == 0) utils::sfread(FLERR,&well[i][0],sizeof(double),3,
                                 fp,NULL,error);
      MPI_Bcast(&well[i][0],3,MPI_DOUBLE,0,world);
    }
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,
                                 fp,NULL,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
/* ----------------------------------------------------------------------
   For new variables
------------------------------------------------------------------------- */
          utils::sfread(FLERR,&gammasb[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&epsilon[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&sigma[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&cfa[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&cfb[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&cfc[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&hth[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&aq[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&bq[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&cq[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&kn[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&kt[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&gamma_n[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&gamma_t[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&xmu[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&hf[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&cut[i][j],sizeof(double),1,fp,NULL,error);
        }
        MPI_Bcast(&gammasb[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cfa[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cfb[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cfc[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&hth[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&aq[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&bq[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cq[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&kn[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&kt[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&gamma_n[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&gamma_t[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&xmu[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&hf[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
/* ---------------------------------------------------------------------- */
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairGayBerne::write_restart_settings(FILE *fp)
{
  fwrite(&gamma,sizeof(double),1,fp);
  fwrite(&upsilon,sizeof(double),1,fp);
  fwrite(&mu,sizeof(double),1,fp);
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);

/* ----------------------------------------------------------------------
   For new flags
------------------------------------------------------------------------- */
  fwrite(&Q_flag,sizeof(int),1,fp);
  fwrite(&C_flag,sizeof(int),1,fp);
  fwrite(&interval,sizeof(int),1,fp);
  fwrite(&A_const,sizeof(int),1,fp);
/* ---------------------------------------------------------------------- */

}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairGayBerne::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    utils::sfread(FLERR,&gamma,sizeof(double),1,fp,NULL,error);
    utils::sfread(FLERR,&upsilon,sizeof(double),1,fp,NULL,error);
    utils::sfread(FLERR,&mu,sizeof(double),1,fp,NULL,error);
    utils::sfread(FLERR,&cut_global,sizeof(double),1,fp,NULL,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,NULL,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,NULL,error);

/* ----------------------------------------------------------------------
   For new flags
------------------------------------------------------------------------- */
    utils::sfread(FLERR,&Q_flag,sizeof(int),1,fp,NULL,error);
    utils::sfread(FLERR,&C_flag,sizeof(int),1,fp,NULL,error);
    utils::sfread(FLERR,&interval,sizeof(int),1,fp,NULL,error);
    utils::sfread(FLERR,&A_const,sizeof(int),1,fp,NULL,error);
/* ---------------------------------------------------------------------- */

  }
  MPI_Bcast(&gamma,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&upsilon,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&mu,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);

/* ----------------------------------------------------------------------
   For new flags
------------------------------------------------------------------------- */
  MPI_Bcast(&Q_flag,1,MPI_INT,0,world);
  MPI_Bcast(&C_flag,1,MPI_INT,0,world);
  MPI_Bcast(&interval,1,MPI_INT,0,world);
  MPI_Bcast(&A_const,1,MPI_INT,0,world);
/* ---------------------------------------------------------------------- */

}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairGayBerne::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    /*fprintf(fp,"%d %g %g %g %g %g %g %g %g\n",i,
            epsilon[i][i],sigma[i][i],*/

/* ----------------------------------------------------------------------
   For new variables
------------------------------------------------------------------------- */
    fprintf(fp,"%d %g %g %g %g %g %g %g %g %g %g %g %g %g 
                %g %g %g %g %g %g %g %g %g\n",i, 
            gammasb[i][i],epsilon[i][i],sigma[i][i],cfa[i][i],cfb[i][i],
            cfc[i][i], hth[i][i], aq[i][i], bq[i][i], cq[i][i], kn[i][i],
            kt[i][i],gamma_n[i][i],gamma_t[i][i], xmu[i][i], hf[i][i], 
/* ---------------------------------------------------------------------- */
            pow(well[i][0],-mu),pow(well[i][1],-mu),pow(well[i][2],-mu),
            pow(well[i][0],-mu),pow(well[i][1],-mu),pow(well[i][2],-mu));
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairGayBerne::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      /*fprintf(fp,"%d %d %g %g %g %g %g %g %g %g %g\n",i,j,
              epsilon[i][i],sigma[i][i],*/

/* ----------------------------------------------------------------------
   For new variables
------------------------------------------------------------------------- */
      fprintf(fp,"%d %d %g %g %g %g %g %g %g %g %g %g %g 
                  %g %g %g %g %g %g %g %g %g %g %g %g\n",i,j, 
              gammasb[i][i],epsilon[i][i],sigma[i][i],cfa[i][i],cfb[i][i],
              cfc[i][i], hth[i][i], aq[i][i], bq[i][i], cq[i][i], kn[i][i], 
/* ---------------------------------------------------------------------- */
              pow(well[i][0],-mu),pow(well[i][1],-mu),pow(well[i][2],-mu),
              pow(well[j][0],-mu),pow(well[j][1],-mu),pow(well[j][2],-mu),
              cut[i][j]);
}

/* ----------------------------------------------------------------------
   compute analytic energy, force (fforce), and torque (ttor & rtor)
   based on rotation matrices a and precomputed matrices b and g
   if newton is off, rtor is not calculated for ghost atoms
------------------------------------------------------------------------- */

/*double PairGayBerne::gayberne_analytic(const int i,const int j,
                                       double a1[3][3],
                                       double a2[3][3], double b1[3][3],
                                       double b2[3][3], double g1[3][3],
                                       double g2[3][3], double *r12,
                                       const double rsq, double *fforce,
                                       double *ttor, double *rtor)*/

/* ----------------------------------------------------------------------
   For new potential function, modelling anisotropic surface charge, 
   and interparticle friction
------------------------------------------------------------------------- */
double PairGayBerne::gayberne_analytic(const int i,const int j,
                                       double a1[3][3],
                                       double a2[3][3], double b1[3][3],
                                       double b2[3][3], double g1[3][3],
                                       double g2[3][3], double *r12,
                                       const double rsq, double *fforce,
                                       double *ttor, double *rtor,
                                       const double iweight, const 
                                       double jweight,
                                       double nveci[3], double nvecj[3],
                                       const double cosi, const double cosj,
                                       int *touch, double *history, 
                                       double *allhistory, const int jj )
/* ---------------------------------------------------------------------- */

{
  double tempv[3], tempv2[3];
  double temp[3][3];
  double temp1,temp2,temp3;

/* ----------------------------------------------------------------------
   Define new variables
------------------------------------------------------------------------- */
  double vr1,vr2,vr3,vnnr,vn1,vn2,vn3,vt1,vt2,vt3, vnn;
  double xtmp,ytmp,ztmp,delx,dely,delz,fx,fy,fz;
  double wr1,wr2,wr3;
  double vtr1,vtr2,vtr3,vrel;
  double shrmag,rsht;
  double mi,mj,meff,damp,ccel,tor1,tor2,tor3;
  double fn,fs,fnn[3],fss[3],fnk[3];
  double xc[3], xca[3], xcb[3], xa[3], xb[3];
  double itor[3], jtor[3];
  double wr[3], iwbody[3], jwbody[3], tmp_iwbody[3], tmp_jwbody[3];
  double radi, radj;
  double inertia[3];
  double rot[3][3];
  double xca_hat[3], xcb_hat[3];
  double nf_torque, f_torque[3];
  double ellipsoid_diameter;
  double cutoff;
/* ---------------------------------------------------------------------- */

  int *type = atom->type;
  int newton_pair = force->newton_pair;
  int nlocal = atom->nlocal;

/* ----------------------------------------------------------------------
   Define new variables
------------------------------------------------------------------------- */
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  double *iquat,*jquat;
  double **x = atom->x;
  double **v = atom->v;
  double *radius = atom->radius;
  double **omega = atom->omega;
  double *rmass = atom->rmass;
  double **angmom = atom->angmom;
  double *drho = atom->drho;
  tagint *tag = atom->tag;
/* ---------------------------------------------------------------------- */

  double r12hat[3];
  MathExtra::normalize3(r12,r12hat);
  double r = sqrt(rsq);

/* ----------------------------------------------------------------------
   Define new variables
------------------------------------------------------------------------- */
  double rinv = 1.0/r;
  double rsqinv = 1.0/rsq;
  int shearupdate = 1;
  if (update->setupflag) shearupdate = 0; 
/* ---------------------------------------------------------------------- */

  // compute distance of closest approach

  double g12[3][3];
  MathExtra::plus3(g1,g2,g12);
  double kappa[3];
  int ierror = MathExtra::mldivide3(g12,r12,kappa);
  if (ierror) error->all(FLERR,"Bad matrix inversion in mldivide3");

  // tempv = G12^-1*r12hat

  tempv[0] = kappa[0]/r;
  tempv[1] = kappa[1]/r;
  tempv[2] = kappa[2]/r;
  double sigma12 = MathExtra::dot3(r12hat,tempv);
  sigma12 = pow(0.5*sigma12,-0.5);
  double h12 = r-sigma12;

/* ----------------------------------------------------------------------
   Cut-off range is modified by a function of h12.
------------------------------------------------------------------------- */
  ellipsoid_diameter = (shape1[type[i]][0]+shape1[type[i]][1]
    +shape1[type[j]][0]+shape1[type[j]][1])/2.0;
  /*ellipsoid_diameter = (shape1[type[i]][0]+shape1[type[i]][2]
    +shape1[type[j]][0]+shape1[type[j]][2])/2.0;*/
  cutoff = sqrt(cutsq[type[i]][type[j]]) - ellipsoid_diameter;
  if(h12 > cutoff){
    fforce[0] = 0.0;
    fforce[1] = 0.0;
    fforce[2] = 0.0;
    ttor[0] = 0.0;
    ttor[1] = 0.0;
    ttor[2] = 0.0;
    rtor[0] = 0.0;
    rtor[1] = 0.0;
    rtor[2] = 0.0;
    return 0.0;
  }
/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   This part is the algorithm to calculate contact and sub-contact 
   points of particles. 
------------------------------------------------------------------------- */
  MathExtra::vecmat(kappa,g1,xca);
  MathExtra::vecmat(kappa,g2,xcb);
  for(int k=0; k<3; k++) xc[k] = xca[k] + x[i][k];
  for(int k=0; k<3; k++) xa[k] = x[i][k] + xca[k]*sigma12/r;
  for(int k=0; k<3; k++) xb[k] = x[j][k] - xcb[k]*sigma12/r;
/* ---------------------------------------------------------------------- */

  // energy
  // compute u_r

  /*double varrho = sigma[type[i]][type[j]]
                  /(h12+gamma*sigma[type[i]][type[j]]);
  double varrho6 = pow(varrho,6.0);
  double varrho12 = varrho6*varrho6;
  double u_r = 4.0*epsilon[type[i]][type[j]]*(varrho12-varrho6);*/

/* ----------------------------------------------------------------------
   This part is the algorithm to calculate energy (u-r) for 
   each interaction (i.e. face-to-face, face-to-edge, 
   edge-to-face, and edge-to-edge)
------------------------------------------------------------------------- */
  double tmp_h12 = h12;

  // face-to-face
  double varrho_cc, varrho6_cc, varrho12_cc, varrho3_cc, u_r_cc;
  varrho_cc   = sigma[type[i]][type[j]]
              /(tmp_h12+gammasb[type[i]][type[j]]*sigma[type[i]][type[j]]);
  varrho12_cc = cfa[type[i]][type[j]]*pow(varrho_cc, 12.0);
  varrho6_cc  = cfb[type[i]][type[j]]*pow(varrho_cc,  6.0);
  varrho3_cc  = cfc[type[i]][type[j]]*pow(varrho_cc,  3.0);
  u_r_cc      = varrho12_cc-varrho6_cc+varrho3_cc;
  // quadratic function
  if( Q_flag == 1 && tmp_h12 < hth[type[i]][type[j]] ){
    u_r_cc = aq[type[i]][type[j]]*tmp_h12*tmp_h12 
           + bq[type[i]][type[j]]*tmp_h12 + cq[type[i]][type[j]];
  }

  // edge-to-edge
  double varrho_hh, varrho6_hh, varrho12_hh, varrho3_hh, u_r_hh;
  varrho_hh   = sigma[3][3]/(tmp_h12+gammasb[3][3]*sigma[3][3]); 
  varrho12_hh = cfa[3][3]*pow(varrho_hh, 12.0);
  varrho6_hh  = cfb[3][3]*pow(varrho_hh,  6.0);
  varrho3_hh  = cfc[3][3]*pow(varrho_hh,  3.0);
  u_r_hh      = varrho12_hh-varrho6_hh+varrho3_hh; 
  // quadratic function
  if( Q_flag == 1 && tmp_h12 < hth[3][3] ){
    u_r_hh = aq[3][3]*tmp_h12*tmp_h12 + bq[3][3]*tmp_h12 + cq[3][3];
  } 

  // face-to-edge
  double varrho_ch, varrho6_ch, varrho12_ch, varrho3_ch, u_r_ch;
  varrho_ch   = sigma[type[i]][3]
              /(tmp_h12+gammasb[type[i]][3]*sigma[type[i]][3]); 
  varrho12_ch = cfa[type[i]][3]*pow(varrho_ch, 12.0);
  varrho6_ch  = cfb[type[i]][3]*pow(varrho_ch,  6.0);
  varrho3_ch  = cfc[type[i]][3]*pow(varrho_ch,  3.0);
  u_r_ch      = varrho12_ch-varrho6_ch+varrho3_ch;
  // quadratic function
  if( Q_flag == 1 && tmp_h12 < hth[type[i]][3] ){
    u_r_ch = aq[type[i]][3]*tmp_h12*tmp_h12 
           + bq[type[i]][3]*tmp_h12 + cq[type[i]][3];
  }

  // edge-to-face
  double varrho_hc, varrho6_hc, varrho12_hc, varrho3_hc, u_r_hc;
  varrho_hc   = sigma[3][type[j]]
              /(tmp_h12+gammasb[3][type[j]]*sigma[3][type[j]]); 
  varrho12_hc = cfa[3][type[j]]*pow(varrho_hc, 12.0);
  varrho6_hc  = cfb[3][type[j]]*pow(varrho_hc,  6.0);
  varrho3_hc  = cfc[3][type[j]]*pow(varrho_hc,  3.0);
  u_r_hc      = varrho12_hc-varrho6_hc+varrho3_hc;
  // quadratic function
  if( Q_flag == 1 && tmp_h12 < hth[3][type[j]] ){
    u_r_hc = aq[3][type[j]]*tmp_h12*tmp_h12 
           + bq[3][type[j]]*tmp_h12 + cq[3][type[j]];
  }

  // total potential energy
  double u_gb = iweight*jweight*u_r_cc
              + iweight*(1.0-jweight)*u_r_ch
              + (1.0-iweight)*jweight*u_r_hc
              + (1.0-iweight)*(1.0-jweight)*u_r_hh;
/* ---------------------------------------------------------------------- */

  // compute eta_12

  double eta = 2.0*lshape[type[i]]*lshape[type[j]];
  double det_g12 = MathExtra::det3(g12);
  eta = pow(eta/det_g12,upsilon);

  // compute chi_12

  double b12[3][3];
  double iota[3];
  MathExtra::plus3(b1,b2,b12);
  ierror = MathExtra::mldivide3(b12,r12,iota);
  if (ierror) error->all(FLERR,"Bad matrix inversion in mldivide3");

  // tempv = G12^-1*r12hat

  tempv[0] = iota[0]/r;
  tempv[1] = iota[1]/r;
  tempv[2] = iota[2]/r;
  double chi = MathExtra::dot3(r12hat,tempv);
  chi = pow(chi*2.0,mu);

  // force
  // compute dUr/dr

  /*temp1 = (2.0*varrho12*varrho-varrho6*varrho)/sigma[type[i]][type[j]];
  temp1 = temp1*24.0*epsilon[type[i]][type[j]];
  double u_slj = temp1*pow(sigma12,3.0)/2.0;
  double dUr[3];
  temp2 = MathExtra::dot3(kappa,r12hat);
  double uslj_rsq = u_slj/rsq;
  dUr[0] = temp1*r12hat[0]+uslj_rsq*(kappa[0]-temp2*r12hat[0]);
  dUr[1] = temp1*r12hat[1]+uslj_rsq*(kappa[1]-temp2*r12hat[1]);
  dUr[2] = temp1*r12hat[2]+uslj_rsq*(kappa[2]-temp2*r12hat[2]);*/

/* ----------------------------------------------------------------------
   This part is the algorithm to calculate interaction forces (dUr/dr) 
   for each interaction (i.e. face-to-face, face-to-edge, 
   edge-to-face, and edge-to-edge)
------------------------------------------------------------------------- */
  //  face-to-face
  double u_slj_cc, dUr_cc[3], uslj_rsq_cc;
  temp1 = (12.0*varrho12_cc*varrho_cc - 6.0*varrho6_cc*varrho_cc 
        + 3.0*varrho3_cc*varrho_cc)/sigma[type[i]][type[j]];
  // quadratic function
  if( Q_flag == 1 && tmp_h12 < hth[type[i]][type[j]] ){
    temp1 = -(2.0*aq[type[i]][type[j]]*tmp_h12+bq[type[i]][type[j]])};
  temp2 = MathExtra::dot3(kappa,r12hat);
  u_slj_cc = temp1*pow(sigma12,3.0)/2.0;
  uslj_rsq_cc = u_slj_cc/rsq;
  dUr_cc[0] = temp1*r12hat[0]+uslj_rsq_cc*(kappa[0]-temp2*r12hat[0]);
  dUr_cc[1] = temp1*r12hat[1]+uslj_rsq_cc*(kappa[1]-temp2*r12hat[1]);
  dUr_cc[2] = temp1*r12hat[2]+uslj_rsq_cc*(kappa[2]-temp2*r12hat[2]);

  // edge-to-edge
  double u_slj_hh, dUr_hh[3], uslj_rsq_hh;
  temp1 = (12.0*varrho12_hh*varrho_hh - 6.0*varrho6_hh*varrho_hh 
        + 3.0*varrho3_hh*varrho_hh)/sigma[3][3];
  // quadratic function
  if( Q_flag == 1 && tmp_h12 < hth[3][3] ){
    temp1 = -(2.0*aq[3][3]*tmp_h12+bq[3][3])};
  temp2 = MathExtra::dot3(kappa,r12hat);
  u_slj_hh = temp1*pow(sigma12,3.0)/2.0;
  uslj_rsq_hh = u_slj_hh/rsq;
  dUr_hh[0] = temp1*r12hat[0]+uslj_rsq_hh*(kappa[0]-temp2*r12hat[0]);
  dUr_hh[1] = temp1*r12hat[1]+uslj_rsq_hh*(kappa[1]-temp2*r12hat[1]);
  dUr_hh[2] = temp1*r12hat[2]+uslj_rsq_hh*(kappa[2]-temp2*r12hat[2]);

  // face-to-edge
  double u_slj_ch, dUr_ch[3], uslj_rsq_ch;
  temp1 = (12.0*varrho12_ch*varrho_ch - 6.0*varrho6_ch*varrho_ch 
        + 3.0*varrho3_ch*varrho_ch)/sigma[type[i]][3];
  // quadratic function
  if( Q_flag == 1 && tmp_h12 < hth[type[i]][3] ){
    temp1 = -(2.0*aq[type[i]][3]*tmp_h12+bq[type[i]][3])};
  temp2 = MathExtra::dot3(kappa,r12hat);
  u_slj_ch = temp1*pow(sigma12,3.0)/2.0;
  uslj_rsq_ch = u_slj_ch/rsq;
  dUr_ch[0] = temp1*r12hat[0]+uslj_rsq_ch*(kappa[0]-temp2*r12hat[0]);
  dUr_ch[1] = temp1*r12hat[1]+uslj_rsq_ch*(kappa[1]-temp2*r12hat[1]);
  dUr_ch[2] = temp1*r12hat[2]+uslj_rsq_ch*(kappa[2]-temp2*r12hat[2]);

  // edge-to-face
  double u_slj_hc, dUr_hc[3], uslj_rsq_hc;
  temp1 = (12.0*varrho12_hc*varrho_hc - 6.0*varrho6_hc*varrho_hc 
        + 3.0*varrho3_hc*varrho_hc)/sigma[3][type[j]];
  // quadratic function
  if( Q_flag == 1 && tmp_h12 < hth[3][type[j]] ){
    temp1 = -(2.0*aq[3][type[j]]*tmp_h12+bq[3][type[j]])};
  temp2 = MathExtra::dot3(kappa,r12hat);
  u_slj_hc = temp1*pow(sigma12,3.0)/2.0;
  uslj_rsq_hc = u_slj_hc/rsq;
  dUr_hc[0] = temp1*r12hat[0]+uslj_rsq_hc*(kappa[0]-temp2*r12hat[0]);
  dUr_hc[1] = temp1*r12hat[1]+uslj_rsq_hc*(kappa[1]-temp2*r12hat[1]);
  dUr_hc[2] = temp1*r12hat[2]+uslj_rsq_hc*(kappa[2]-temp2*r12hat[2]);

  // cpmpute diweight/dr and djweight/dr

  double dot_ni_r12hat = MathExtra::dot3(nveci,r12hat);
  double norm_ni = sqrt( MathExtra::dot3(nveci,nveci) );
  double diweight[3];
  if( cosi >=0.0 ){
    diweight[0] = (2.0*A_const)*pow(cosi,(2.0*A_const)-1.0)
                /(norm_ni*rsq)*(nveci[0]-dot_ni_r12hat*r12hat[0]);
    diweight[1] = (2.0*A_const)*pow(cosi,(2.0*A_const)-1.0)
                /(norm_ni*rsq)*(nveci[1]-dot_ni_r12hat*r12hat[1]);
    diweight[2] = (2.0*A_const)*pow(cosi,(2.0*A_const)-1.0)
                /(norm_ni*rsq)*(nveci[2]-dot_ni_r12hat*r12hat[2]);
  }else if( cosi < 0.0 ){
    diweight[0] = -(2.0*A_const)*pow(-cosi,(2.0*A_const)-1.0)
                /(norm_ni*rsq)*(nveci[0]-dot_ni_r12hat*r12hat[0]);
    diweight[1] = -(2.0*A_const)*pow(-cosi,(2.0*A_const)-1.0)
                /(norm_ni*rsq)*(nveci[1]-dot_ni_r12hat*r12hat[1]);
    diweight[2] = -(2.0*A_const)*pow(-cosi,(2.0*A_const)-1.0)
                /(norm_ni*rsq)*(nveci[2]-dot_ni_r12hat*r12hat[2]);
  }

  double dot_nj_r12hat = MathExtra::dot3(nvecj,r12hat);
  double norm_nj = sqrt( MathExtra::dot3(nvecj,nvecj) );
  double djweight[3];
  if( cosj >= 0.0 ){
    djweight[0] = (2.0*A_const)*pow(cosj,(2.0*A_const)-1.0)
                /(norm_nj*rsq)*(nvecj[0]-dot_nj_r12hat*r12hat[0]);
    djweight[1] = (2.0*A_const)*pow(cosj,(2.0*A_const)-1.0)
                /(norm_nj*rsq)*(nvecj[1]-dot_nj_r12hat*r12hat[1]);
    djweight[2] = (2.0*A_const)*pow(cosj,(2.0*A_const)-1.0)
                /(norm_nj*rsq)*(nvecj[2]-dot_nj_r12hat*r12hat[2]);
  }else if( cosj < 0.0 ){
    djweight[0] = -(2.0*A_const)*pow(-cosj,(2.0*A_const)-1.0)
                /(norm_nj*rsq)*(nvecj[0]-dot_nj_r12hat*r12hat[0]);
    djweight[1] = -(2.0*A_const)*pow(-cosj,(2.0*A_const)-1.0)
                /(norm_nj*rsq)*(nvecj[1]-dot_nj_r12hat*r12hat[1]);
    djweight[2] = -(2.0*A_const)*pow(-cosj,(2.0*A_const)-1.0)
                /(norm_nj*rsq)*(nvecj[2]-dot_nj_r12hat*r12hat[2]);
  }

  // compute dUr/dr
  double dUr[3];
  dUr[0] = iweight*jweight*dUr_cc[0] 
         + iweight*djweight[0]*u_r_cc 
         + diweight[0]*jweight*u_r_cc
         + iweight*(1.0-jweight)*dUr_ch[0] 
         - iweight*djweight[0]*u_r_ch 
         + diweight[0]*(1.0-jweight)*u_r_ch
         + (1.0-iweight)*jweight*dUr_hc[0] 
         + (1.0-iweight)*djweight[0]*u_r_hc 
         - diweight[0]*jweight*u_r_hc
         + (1.0-iweight)*(1.0-jweight)*dUr_hh[0] 
         - (1.0-iweight)*djweight[0]*u_r_hh 
         - diweight[0]*(1.0-jweight)*u_r_hh;
  dUr[1] = iweight*jweight*dUr_cc[1] 
         + iweight*djweight[1]*u_r_cc 
         + diweight[1]*jweight*u_r_cc
         + iweight*(1.0-jweight)*dUr_ch[1] 
         - iweight*djweight[1]*u_r_ch 
         + diweight[1]*(1.0-jweight)*u_r_ch
         + (1.0-iweight)*jweight*dUr_hc[1] 
         + (1.0-iweight)*djweight[1]*u_r_hc 
         - diweight[1]*jweight*u_r_hc
         + (1.0-iweight)*(1.0-jweight)*dUr_hh[1] 
         - (1.0-iweight)*djweight[1]*u_r_hh 
         - diweight[1]*(1.0-jweight)*u_r_hh;
  dUr[2] = iweight*jweight*dUr_cc[2] 
         + iweight*djweight[2]*u_r_cc 
         + diweight[2]*jweight*u_r_cc
         + iweight*(1.0-jweight)*dUr_ch[2] 
         - iweight*djweight[2]*u_r_ch 
         + diweight[2]*(1.0-jweight)*u_r_ch
         + (1.0-iweight)*jweight*dUr_hc[2] 
         + (1.0-iweight)*djweight[2]*u_r_hc 
         - diweight[2]*jweight*u_r_hc
         + (1.0-iweight)*(1.0-jweight)*dUr_hh[2] 
         - (1.0-iweight)*djweight[2]*u_r_hh 
         - diweight[2]*(1.0-jweight)*u_r_hh;
/* ---------------------------------------------------------------------- */

  // compute dChi_12/dr

  double dchi[3];
  temp1 = MathExtra::dot3(iota,r12hat);
  temp2 = -4.0/rsq*mu*pow(chi,(mu-1.0)/mu);
  dchi[0] = temp2*(iota[0]-temp1*r12hat[0]);
  dchi[1] = temp2*(iota[1]-temp1*r12hat[1]);
  dchi[2] = temp2*(iota[2]-temp1*r12hat[2]);

  temp1 = -eta*u_gb;
  temp3 = eta*chi;
  fforce[0] = temp1*dchi[0]-temp3*dUr[0];
  fforce[1] = temp1*dchi[1]-temp3*dUr[1];
  fforce[2] = temp1*dchi[2]-temp3*dUr[2];

  // torque for particle 1 and 2
  // compute dUr

  /*tempv[0] = -uslj_rsq*kappa[0];
  tempv[1] = -uslj_rsq*kappa[1];
  tempv[2] = -uslj_rsq*kappa[2];*/

/* ----------------------------------------------------------------------
   Modified torque calculation
------------------------------------------------------------------------- */
  double tempv[0] = -(1.0/r)*(iweight*jweight*u_slj_cc/r*kappa[0] 
           + iweight*djweight[0]*u_r_cc 
           + diweight[0]*jweight*u_r_cc
           + iweight*(1.0-jweight)*u_slj_ch/r*kappa[0] 
           - iweight*djweight[0]*u_r_ch 
           + diweight[0]*(1.0-jweight)*u_r_ch
           + (1.0-iweight)*jweight*u_slj_hc/r*kappa[0] 
           + (1.0-iweight)*djweight[0]*u_r_hc 
           - diweight[0]*jweight*u_r_hc
           + (1.0-iweight)*(1.0-jweight)*u_slj_hh/r*kappa[0] 
           - (1.0-iweight)*djweight[0]*u_r_hh 
           - diweight[0]*(1.0-jweight)*u_r_hh);
  tempv[1] = -(1.0/r)*(iweight*jweight*u_slj_cc/r*kappa[1] 
           + iweight*djweight[1]*u_r_cc 
           + diweight[1]*jweight*u_r_cc
           + iweight*(1.0-jweight)*u_slj_ch/r*kappa[1] 
           - iweight*djweight[1]*u_r_ch 
           + diweight[1]*(1.0-jweight)*u_r_ch
           + (1.0-iweight)*jweight*u_slj_hc/r*kappa[1] 
           + (1.0-iweight)*djweight[1]*u_r_hc 
           - diweight[1]*jweight*u_r_hc
           + (1.0-iweight)*(1.0-jweight)*u_slj_hh/r*kappa[1] 
           - (1.0-iweight)*djweight[1]*u_r_hh 
           - diweight[1]*(1.0-jweight)*u_r_hh);
  tempv[2] = -(1.0/r)*(iweight*jweight*u_slj_cc/r*kappa[2] 
           + iweight*djweight[2]*u_r_cc 
           + diweight[2]*jweight*u_r_cc
           + iweight*(1.0-jweight)*u_slj_ch/r*kappa[2] 
           - iweight*djweight[2]*u_r_ch 
           + diweight[2]*(1.0-jweight)*u_r_ch
           + (1.0-iweight)*jweight*u_slj_hc/r*kappa[2] 
           + (1.0-iweight)*djweight[2]*u_r_hc 
           - diweight[2]*jweight*u_r_hc
           + (1.0-iweight)*(1.0-jweight)*u_slj_hh/r*kappa[2] 
           - (1.0-iweight)*djweight[2]*u_r_hh 
           - diweight[2]*(1.0-jweight)*u_r_hh);
/* ---------------------------------------------------------------------- */

  MathExtra::vecmat(kappa,g1,tempv2);
  MathExtra::cross3(tempv,tempv2,dUr);
  double dUr2[3];

  if (newton_pair || j < nlocal) {
    MathExtra::vecmat(kappa,g2,tempv2);
    MathExtra::cross3(tempv,tempv2,dUr2);
  }


  // compute d_chi

  MathExtra::vecmat(iota,b1,tempv);
  MathExtra::cross3(tempv,iota,dchi);
  dchi[0] *= temp2;
  dchi[1] *= temp2;
  dchi[2] *= temp2;
  double dchi2[3];

  if (newton_pair || j < nlocal) {
    MathExtra::vecmat(iota,b2,tempv);
    MathExtra::cross3(tempv,iota,dchi2);
    dchi2[0] *= temp2;
    dchi2[1] *= temp2;
    dchi2[2] *= temp2;
  }

  // compute d_eta

  double deta[3];
  deta[0] = deta[1] = deta[2] = 0.0;
  compute_eta_torque(g12,a1,shape2[type[i]],temp);
  temp1 = -eta*upsilon;
  for (int m = 0; m < 3; m++) {
    for (int y = 0; y < 3; y++) tempv[y] = temp1*temp[m][y];
    MathExtra::cross3(a1[m],tempv,tempv2);
    deta[0] += tempv2[0];
    deta[1] += tempv2[1];
    deta[2] += tempv2[2];
  }

  // compute d_eta for particle 2

  double deta2[3];
  if (newton_pair || j < nlocal) {
    deta2[0] = deta2[1] = deta2[2] = 0.0;
    compute_eta_torque(g12,a2,shape2[type[j]],temp);
    for (int m = 0; m < 3; m++) {
      for (int y = 0; y < 3; y++) tempv[y] = temp1*temp[m][y];
      MathExtra::cross3(a2[m],tempv,tempv2);
      deta2[0] += tempv2[0];
      deta2[1] += tempv2[1];
      deta2[2] += tempv2[2];
    }
  }

  // torque
  temp1 = u_gb*eta;
  temp2 = u_gb*chi;
  temp3 = chi*eta;

  ttor[0] = (temp1*dchi[0]+temp2*deta[0]+temp3*dUr[0]) * -1.0;
  ttor[1] = (temp1*dchi[1]+temp2*deta[1]+temp3*dUr[1]) * -1.0;
  ttor[2] = (temp1*dchi[2]+temp2*deta[2]+temp3*dUr[2]) * -1.0;

  if (newton_pair || j < nlocal) {
    rtor[0] = (temp1*dchi2[0]+temp2*deta2[0]+temp3*dUr2[0]) * -1.0;
    rtor[1] = (temp1*dchi2[1]+temp2*deta2[1]+temp3*dUr2[1]) * -1.0;
    rtor[2] = (temp1*dchi2[2]+temp2*deta2[2]+temp3*dUr2[2]) * -1.0;
  }

  /*return temp1*chi;*/

/* ----------------------------------------------------------------------
   Thia part is for calculation of friction. Friction is calculated using Hookean contact model. 
------------------------------------------------------------------------- */
  double vr[3], vn[3], vt[3], vnr[3], vrr[3];

  /*zero set*/
  fss[0] = 0.0;
  fss[1] = 0.0;
  fss[2] = 0.0;
  fnn[0] = fforce[0];
  fnn[1] = fforce[1];
  fnn[2] = fforce[2];
  
  /* unset non-touching neighbors*/
  if( h12 > hf[type[i]][type[j]] ){
    touch[jj] = 0;
    history = &allhistory[size_history*jj];
    for (int k = 0; k < size_history; k++) history[k] = 0.0;
  }else{
  /* calculate tangential force */
    /* relative translational velocity */
    vr[0] = v[i][0] - v[j][0];
    vr[1] = v[i][1] - v[j][1];
    vr[2] = v[i][2] - v[j][2];
    /* normal component */
    vnnr = MathExtra::dot3(vr,r12hat);
    vn[0] = vnnr*r12hat[0];
    vn[1] = vnnr*r12hat[1];
    vn[2] = vnnr*r12hat[2];
    /* tangential component */
    vt[0] = vr[0] - vn[0];
    vt[1] = vr[1] - vn[1];
    vt[2] = vr[2] - vn[2];

    /* principal moments of inertia */
    inertia[0] = rmass[i] * (shape1[type[i]][1]*shape1[type[i]][1]
                            +shape1[type[i]][2]*shape1[type[i]][2]) / 5.0;
    inertia[1] = rmass[i] * (shape1[type[i]][0]*shape1[type[i]][0]
                            +shape1[type[i]][2]*shape1[type[i]][2]) / 5.0;
    inertia[2] = rmass[i] * (shape1[type[i]][0]*shape1[type[i]][0]
                            +shape1[type[i]][1]*shape1[type[i]][1]) / 5.0;

    /*angular velocity:wbody = angular velocity in xyz coordination system*/
    iquat = bonus[ellipsoid[i]].quat;
    MathExtra::quat_to_mat(iquat,rot);
    MathExtra::transpose_matvec(rot,angmom[i],tmp_iwbody);
    tmp_iwbody[0] /= inertia[0];
    tmp_iwbody[1] /= inertia[1];
    tmp_iwbody[2] /= inertia[2];
    MathExtra::matvec(rot,tmp_iwbody,iwbody);

    /* principal moments of inertia */
    inertia[0] = rmass[j] * (shape1[type[j]][1]*shape1[type[j]][1]
                            +shape1[type[j]][2]*shape1[type[j]][2]) / 5.0;
    inertia[1] = rmass[j] * (shape1[type[j]][0]*shape1[type[j]][0]
                            +shape1[type[j]][2]*shape1[type[j]][2]) / 5.0;
    inertia[2] = rmass[j] * (shape1[type[j]][0]*shape1[type[j]][0]
                            +shape1[type[j]][1]*shape1[type[j]][1]) / 5.0;

    /*angular velocity:wbody = angular velocity in xyz coordination system*/
    jquat = bonus[ellipsoid[j]].quat;
    MathExtra::quat_to_mat(jquat,rot);
    MathExtra::transpose_matvec(rot,angmom[j],tmp_jwbody);
    tmp_jwbody[0] /= inertia[0];
    tmp_jwbody[1] /= inertia[1];
    tmp_jwbody[2] /= inertia[2];
    MathExtra::matvec(rot,tmp_jwbody,jwbody);

    /* contacr point */
    xca[0]=x[i][0]-xa[0];
    xca[1]=x[i][1]-xa[1];
    xca[2]=x[i][2]-xa[2];
    xcb[0]=xb[0]-x[j][0];
    xcb[1]=xb[1]-x[j][1];
    xcb[2]=xb[2]-x[j][2];
    
    /*relative rotational velocity*/
    vrr[0] = (xca[2]*iwbody[1]-xca[1]*iwbody[2]) 
           + (xcb[2]*jwbody[1]-xcb[1]*jwbody[2]);
    vrr[1] = (xca[0]*iwbody[2]-xca[2]*iwbody[0]) 
           + (xcb[0]*jwbody[2]-xcb[2]*jwbody[0]);
    vrr[2] = (xca[1]*iwbody[0]-xca[0]*iwbody[1]) 
           + (xcb[1]*jwbody[0]-xcb[0]*jwbody[1]);

    /* normal component */
    vnnr = MathExtra::dot3(vrr,r12hat);
    vnr[0] = vnnr*r12hat[0];
    vnr[1] = vnnr*r12hat[1];
    vnr[2] = vnnr*r12hat[2];
    /* tangential component */
    vt[0] -= vrr[0] - vnr[0];
    vt[1] -= vrr[1] - vnr[1];
    vt[2] -= vrr[2] - vnr[2];
    
    /* shear history effects */
    touch[jj] = 1;
    history = &allhistory[size_history*jj];
    if(shearupdate){
      history[0] += vt[0]*dt;
      history[1] += vt[1]*dt;
      history[2] += vt[2]*dt;
    }
    shrmag = sqrt(history[0]*history[0] + history[1]*history[1] 
                + history[2]*history[2]);

    /* rotate shear displacements */
    rsht = MathExtra::dot3(history,r12hat);
    if(shearupdate){
      history[0] -= rsht*r12hat[0];
      history[1] -= rsht*r12hat[1];
      history[2] -= rsht*r12hat[2];
    }

    // mean mass : meff = effective mass of pair of particles */
    mi = rmass[i];
    mj = rmass[j];
    meff = mi*mj / (mi+mj);

    /* tangential forces = shear + tangential velocity damping */
    fss[0] = - (kt[type[i]][type[j]]*history[0] 
             + meff*gamma_t[type[i]][type[j]]*vt[0]);
    fss[1] = - (kt[type[i]][type[j]]*history[1] 
             + meff*gamma_t[type[i]][type[j]]*vt[1]);
    fss[2] = - (kt[type[i]][type[j]]*history[2] 
             + meff*gamma_t[type[i]][type[j]]*vt[2]);

    /* rescale frictional displacements and forces if needed */
    fs = sqrt(fss[0]*fss[0] + fss[1]*fss[1] + fss[2]*fss[2]);

    /* Hookean contacr model */
    fn = MathExtra::dot3(fforce,r12hat);
    fn = sqrt(fn*fn);
    fn *= xmu[type[i]][type[j]];

    if (fabs(fs) > fn) {
      if (shrmag != 0.0) {
        history[0] = (fn/fs) * (history[0] + meff*gamma_t[type[i]][type[j]]
                               *vt[0]/kt[type[i]][type[j]]) 
                - meff*gamma_t[type[i]][type[j]]*vt[0]/kt[type[i]][type[j]];
        history[1] = (fn/fs) * (history[1] + meff*gamma_t[type[i]][type[j]]
                               *vt[1]/kt[type[i]][type[j]]) 
                - meff*gamma_t[type[i]][type[j]]*vt[1]/kt[type[i]][type[j]];
        history[2] = (fn/fs) * (history[2] + meff*gamma_t[type[i]][type[j]]
                               *vt[2]/kt[type[i]][type[j]]) 
                - meff*gamma_t[type[i]][type[j]]*vt[2]/kt[type[i]][type[j]];
        fss[0] *= fn/fs;
        fss[1] *= fn/fs;
        fss[2] *= fn/fs;
      } else{
        fss[0] = 0.0;
        fss[1] = 0.0;
        fss[2] = 0.0;
      }
    }

    /* force */
    fforce[0] += fss[0];
    fforce[1] += fss[1];
    fforce[2] += fss[2];
    
    /* torque */
    f_torque[0] = fss[0];
    f_torque[1] = fss[1];
    f_torque[2] = fss[2];
    MathExtra::vecmat(kappa,g1,xca);
    MathExtra::cross3(xca,f_torque,itor);
    ttor[0] += itor[0];
    ttor[1] += itor[1];
    ttor[2] += itor[2];
    if (newton_pair || j < nlocal) {
      f_torque[0] = fss[0];
      f_torque[1] = fss[1];
      f_torque[2] = fss[2];
      MathExtra::vecmat(kappa,g2,xcb);
      MathExtra::cross3(xcb,f_torque,jtor);
      rtor[0] += jtor[0];
      rtor[1] += jtor[1];
      rtor[2] += jtor[2];
    }
  }
/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Thia part is for debug.
   C_flag = 0: non-active
   C_flag = 1: active
------------------------------------------------------------------------- */
  if( C_flag == 1 && ntimestep%interval == 0.0 ){
    FILE *fp;
    char filename[30];
    sprintf(filename, "contact_debug_%d.txt", ntimestep);
    fp = fopen(filename,"a");
    fprintf(fp, "%e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n", 
                 x[i][0], x[i][1], x[i][2], x[j][0], x[j][1], x[j][2], 
                 fnn[0], fnn[1], fnn[2], fss[0], fss[1], fss[2], 
                 history[0], history[1], history[2], h12);
    fclose(fp);
  }
/* ---------------------------------------------------------------------- */
  /* potential energy */
  double pe = u_gb*eta*chi;
  return pe;
}
//__________________________________________________________________