/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(gayberne,PairGayBerne)

#else

#ifndef LMP_PAIR_GAYBERNE_H
#define LMP_PAIR_GAYBERNE_H

#include "pair.h"

/*--------------- Modified by Yohei Nakamichi ---------------*/
#include<stdio.h>
#include<math.h>
#include<errno.h>
#include<stdlib.h>
#include<complex>
using namespace std;
/*-----------------------------------------------------------*/

namespace LAMMPS_NS {

class PairGayBerne : public Pair {
 public:
  PairGayBerne(LAMMPS *lmp);
  virtual ~PairGayBerne();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  virtual void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);

 protected:
  enum{SPHERE_SPHERE,SPHERE_ELLIPSE,ELLIPSE_SPHERE,ELLIPSE_ELLIPSE};

  double cut_global;
  double **cut;

  double gamma,upsilon,mu;   // Gay-Berne parameters
  double **shape1;           // per-type radii in x, y and z
  double **shape2;           // per-type radii in x, y and z SQUARED
  double *lshape;            // precalculation based on the shape
  double **well;             // well depth scaling along each axis ^ -1.0/mu
  double **epsilon,**sigma;  // epsilon and sigma values for atom-type pairs
  /*--------------- Modified by Yohei Nakamichi ---------------*/
  double **gammasb;    // gammasb values for atom-type pairs
  double **cfa, **cfb, **cfc;
  double **kn, **kt, **gamma_n, **gamma_t, **xmu; //friction model parameters
  int    DEM_flag; // if this flag is one, normal force is calculated using spring model.
  /* for calculation of closest distance */
  double l1i[3],l2i[3],m1i[3],m2i[3],n1i[3],n2i[3],di[3]; //Input vectors
  double A1,A2,B1,B2,C1,C2; //Input semiaxes lenghts
  double d[3],p0[3],p[3],s[3],l1[3],l2[3],m1[3],m2[3],n1[3],n2[3],dxp0[3]; //Normalized vectors
  double k1[3], k2[3], kn1[3], kn2[3], xc1[3], xc2[3], xt1[3], xt2[3];
  #define pi 4*atan(1.0)
  #define gratio (1.0+sqrt(5.0))*0.5
  #define o1g 1.0/(1.0+gratio)
  //#define tolerance 1E-9
  #define tolerance 1E-7
  #define delt 1E-5
  /* quadratic function for Born repulsion */
  int    Q_flag;
  double **aq, **bq, **cq, **hth; // parameters for quadratic finction
  /* output contact information */
  int C_flag;
  int interval;
  /* set threshhold of friction */
  double **hf;
  /*-----------------------------------------------------------*/

  int **form;
  double **lj1,**lj2,**lj3,**lj4;
  double **offset;
  int *setwell;
  class AtomVecEllipsoid *avec;

  void allocate();
  /*double gayberne_analytic(const int i, const int j, double a1[3][3],
                           double a2[3][3], double b1[3][3], double b2[3][3],
                           double g1[3][3], double g2[3][3], double *r12,
                           const double rsq, double *fforce, double *ttor,
                           double *rtor);*/
  /*--------------- Modified by Yohei Nakamichi ---------------*/
  double gayberne_analytic(const int i, const int j, double a1[3][3],
                           double a2[3][3], double b1[3][3], double b2[3][3],
                           double g1[3][3], double g2[3][3], double *r12,
                           const double rsq, double *fforce, double *ttor,
                           double *rtor,
                           const double iweight, const double jweight,
                           double nveci[3], double nvecj[3],
                           const double cosi, const double cosj,
                           int *touch, double *history, double *allhistory, const int jj);
  /*-----------------------------------------------------------*/
  double gayberne_lj(const int i, const int j, double a1[3][3],
                     double b1[3][3],double g1[3][3],double *r12,
                     const double rsq, double *fforce, double *ttor);
  void compute_eta_torque(double m[3][3], double m2[3][3],
                          double *s, double ans[3][3]);

  /*--------------- Modified by Yohei Nakamichi ---------------*/
  void crossP(double *x,double *y,double *z);
  void norm(double *vec,double *nvec);
  double mag(double *V);
  double dotP(double *Vec1,double *Vec2);
  double plane_int(double); //
  double distance2d(double,double,double,double,double,double); //
  complex<double> c_cbrt( complex<double> ); //
  double ellipsoids(void);
  /*-----------------------------------------------------------*/

  /*--------------- Modified by Yohei Nakamichi ---------------*/
  int use_history;
  class FixNeighHistory *fix_history;
  int size_history;
  double dt;
  int ntimestep;
  /*-----------------------------------------------------------*/

};

}
#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair gayberne requires atom style ellipsoid

Self-explanatory.

E: Pair gayberne requires atoms with same type have same shape

Self-explanatory.

E: Pair gayberne epsilon a,b,c coeffs are not all set

Each atom type involved in pair_style gayberne must
have these 3 coefficients set at least once.

E: Bad matrix inversion in mldivide3

This error should not occur unless the matrix is badly formed.

*/
