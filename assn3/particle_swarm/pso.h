#ifndef _PSO_H_
#define _PSO_H_

/* Spit out debug info. 
 * FIXME: comment out when measuring execution time 
 * */
// #define VERBOSE_DEBUG
// #define SIMPLE_DEBUG

/* Structure of a particle */
typedef struct particle_s {
    int dim;            /* Dimension of particle */
    float *x;           /* Particle position */
    float *v;           /* Particle velocity */
    float *pbest;       /* Best particle position seen */
    float fitness;      /* Fitness of particle */
    int g;              /* Index of best performing particle in swarm */
} particle_t;

/* Structure of the swarm */
typedef struct swarm_s {
    int num_particles;          /* Number of particles */
    particle_t *particle;       /* Particle within swarm */
} swarm_t;

/* Function prototypes */
void print_args(char *, int, int, float, float);
void pso_print_swarm(swarm_t *);
void pso_print_particle(particle_t *);
float uniform(float, float);
swarm_t *pso_init(char *, int, int, float, float);
int pso_eval_fitness(char *, particle_t *, float *);
int pso_solve_gold(char *, swarm_t *, float, float, int);
void pso_free(swarm_t *);
int pso_get_best_fitness(swarm_t *);
int optimize_gold(char *, int, int, float, float, int);
int optimize_using_omp(char *, int, int, float, float, int, int);

/* Optimization test functions */
float pso_eval_rastrigin(particle_t *);
float pso_eval_booth(particle_t *);
float pso_eval_holder_table(particle_t *);
float pso_eval_eggholder(particle_t *);
float pso_eval_schwefel(particle_t *);

#endif /* _PSO_H_ */
