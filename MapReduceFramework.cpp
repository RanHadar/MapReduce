#include "MapReduceClient.h"
#include "MapReduceFramework.h"
#include <pthread.h>
#include <vector>
#include <queue>
#include <malloc.h>
#include <atomic>
#include <iostream>
#include <functional>
#include <algorithm>
#include "Barrier.h"
#include <semaphore.h>
#include <unistd.h>

using namespace std;

//this struct is used to monitor wait for job
struct WaitJobMutex{
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
};

struct JobContext;

//this struct describe each thread
struct threadContext {
    pthread_t thread;
    IntermediateVec vec;
    JobContext* job;
};

struct JobContext{

    //queue for all the  Intermediate Vecs that passed the shuffle phase
    queue<IntermediateVec*>*  shuffled{};

    //a semaphore used on the reduce phase
    sem_t sem{};

    //the Job State
    JobState* state{};

    //the input vec
    const InputVec* inputVec{};

    //the client
    const MapReduceClient* client{};

    //flag thats true iff no thread entered the shuffle phase
    bool needToShuff = true;

    //barrier used before entering the reduce phase
    Barrier* barrier{};

    //mutex that protects the the thread that enters the shuffle phase ensures only one enters
    pthread_mutex_t  shuff_mutex = PTHREAD_MUTEX_INITIALIZER;

    //mutex that  responsible for the wait for job
    WaitJobMutex jobMutex;

    //true iff the shuffle phase finished
    bool shuffleFinished = false;

    //mutex that  protects the shuffle queue
    pthread_mutex_t queue_mutex  = PTHREAD_MUTEX_INITIALIZER;

    //mutex that  protects the emit3
    pthread_mutex_t emit3_mutex  = PTHREAD_MUTEX_INITIALIZER;

    //vector for all threads
    vector<threadContext>* threads{};

    //the outout vec
    OutputVec *outVec{};

    //counter to keep track of how much keys were  processed
    atomic<unsigned int> atomic_counter{0};

    //    //counter to keep track of how much input finished map phase
    atomic<unsigned int> precentege_atomic_counter{0};

    //how many keys are in the reduce phase total
    unsigned long reduce_sum =0;
};


void emit2 (K2* key, V2* value, void* context){
    auto  vec = (IntermediateVec*)(context);
    vec->emplace_back(key,value);
}


void emit3 (K3* key, V3* value, void* context){

    auto job = (JobContext*)context;
    if ( pthread_mutex_lock( &job->emit3_mutex ) != 0){
        fprintf(stderr, "error on pthread_mutex_lock");
        exit(1);
    }

    job->outVec->emplace_back(key, value);

    if (  pthread_mutex_unlock( &job->emit3_mutex) != 0){
        fprintf(stderr, "error on pthread_mutex_lock");
        exit(1);
    }

}

void* runJobThread(void *pContext);


JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel){
    auto  job = new JobContext();
    job->state = new JobState();
    job ->state->stage = UNDEFINED_STAGE;
    job -> state -> percentage = 0.0;
    job -> client = &client;
    job -> inputVec = &inputVec;
    job->outVec = &outputVec;
    job->threads = new vector<threadContext>(multiThreadLevel);
    job->barrier = new Barrier(multiThreadLevel);
    sem_init(&job->sem,0,0);
    job->shuffled = new queue<IntermediateVec*>();
    for (int i = 0; i < multiThreadLevel;i++){
        job->threads->at(i).job=job;
       if( pthread_create(&job->threads->at(i).thread,nullptr,runJobThread,&job->threads->at(i)) !=0){
            fprintf(stderr, "error on pthread_create");
            exit(1);
        }
    }
    for (int i = 0; i < multiThreadLevel;i++)   pthread_join(job->threads->at(i).thread,nullptr);

    return job;
}


void shuffle(JobContext* job);

//this method do the job for each threads=
void* runJobThread(void *pContext){
    auto  thread = (threadContext*)pContext;
    JobContext* job = thread->job;
    job->state->stage = MAP_STAGE;
    unsigned int old_value = (job->atomic_counter)++;
    while(old_value<job->inputVec->size()) {
        const InputPair pair =job->inputVec->at(old_value); //get the next input at old value
        job->client->map(pair.first , pair.second ,&thread->vec); //add the output of map to the thread->vec
        old_value =(job->atomic_counter)++;
        job->precentege_atomic_counter++;
    }

    if(!thread->vec.empty()){ //sort the map phase output vector for each thread
        sort(thread->vec.begin(),  thread->vec.end(), [](IntermediatePair &p1, IntermediatePair &p2)  {
            return *(p1.first) < *(p2.first);

        });
    }

    job->barrier->barrier();
    job->state->stage=REDUCE_STAGE;

    //decide which thread needs to shuffle
    if (  pthread_mutex_lock(&job->shuff_mutex)!= 0){
        fprintf(stderr, "error on pthread_mutex_lock");
        exit(1);
    }
    if(job->needToShuff){
        job -> needToShuff = false;
        for (auto &threads : *job->threads) { //count how many keys are in the reduce phase total
            job->reduce_sum+=threads.vec.size();
        }
        job->atomic_counter = 0; //reset the progress counter for the reduce phase
            if (pthread_mutex_unlock(&job->shuff_mutex) != 0){
            fprintf(stderr, "error on pthread_mutex_unlock");
            exit(1);
        }

        shuffle(job); //start shuffle
    } else{
        if (pthread_mutex_unlock(&job->shuff_mutex) != 0){
            fprintf(stderr, "error on pthread_mutex_unlock");
            exit(1);
        }
    }

    //as long as we didn't finish with all the shuffled vectors continue to reduce
    while(!(job->shuffled->empty())||!(job->shuffleFinished)) {
        if (sem_wait(&job->sem) < 0) {
            fprintf(stderr, "error on sem_wait");
            exit(1);
        }
        if (pthread_mutex_lock(&job->queue_mutex) != 0) {
            fprintf(stderr, "error on pthread_mutex_lock");
            exit(1);
        }
        // if by the time we got here all the shuffled vectors were processed we need to exit
        if (job->shuffled->empty())
        {
            if (pthread_mutex_unlock(&job->queue_mutex) != 0) {
                fprintf(stderr, "error on pthread_mutex_unlock");
                exit(1);
            }
            break;
        }
        //take one vector from the shuffled ones
        IntermediateVec* toReduce;
        toReduce = job->shuffled->front();
        job->shuffled->pop();

        if (pthread_mutex_unlock(&job->queue_mutex) != 0) {
            fprintf(stderr, "error on pthread_mutex_unlock");
            exit(1);
        }

        job->client->reduce(toReduce, job);
        job->atomic_counter+=toReduce->size(); //update we processed another vector
        delete (toReduce);
    }

    //broadcast that job is finished
    if (pthread_cond_broadcast(&job->jobMutex.cv) != 0){
        fprintf(stderr, "error on pthread_cond_broadcast");
        exit(1);
    }
    return job;
  }


K2 *findLargest(JobContext *pContext);


//this method do the shuffle phase
void shuffle(JobContext* job) {

    K2* largest = findLargest(job);

    while (largest != nullptr){ //as long as there are keys do:

        auto   vec = new IntermediateVec();
            for (auto &threads : *job->threads) {
            //we take out all the keys that are equal to the largest key amd push them to new vector
            while(!(threads.vec.empty())&&
            !(*(threads.vec.back().first)<(*largest))){//we dont have  a = so we did <= and its the big
                auto pair = threads.vec.back();
                threads.vec.pop_back();
                vec->emplace_back(pair.first, pair.second);

            }
        }

        if (pthread_mutex_lock(&job->queue_mutex) != 0){
            fprintf(stderr, "error on pthread_mutex_lock");
            exit(1);
        }

        //push the bew vector to the queue
        job->shuffled->push(vec);
        largest = findLargest(job);

        if(largest== nullptr){ //we enter if there are no more keys to process
            job->shuffleFinished = true;

            //update the semaphore by the number of thread so no one will get stuck
            for(unsigned int i = 0; i<job->threads->size();i++)
                if(sem_post(&job->sem)<0){
                    fprintf(stderr, "error on sem_post");
                    exit(1);
                }
        }else {
            if (sem_post(&job->sem) < 0) {
                fprintf(stderr, "error on sem_post");
                exit(1);

            }
        }
        if (pthread_mutex_unlock(&job->queue_mutex) != 0){
            fprintf(stderr, "error on pthread_mutex_unlock");
            exit(1);
        }

    }
}

//this method returns the largest key of all the remaining intemidient vecs
K2 *findLargest(JobContext *pContext) {
    unsigned long  sizeVec = pContext->threads->size();
    unsigned int i;
    //skipped all the empty vecs
    for (i = 0; i < sizeVec; i++){
        if(!(pContext->threads->at(i).vec.empty())){
            break;
        }
    }

    if (i == sizeVec){ //if all vecs are empty
        return nullptr;
    }
    //get the largest
    K2 * largest = (pContext->threads->at(i).vec.back().first);
    for (; i < sizeVec; i++){
        if((!(pContext->threads->at(i).vec.empty()))
                   &&((*largest)<(*pContext->threads->at(i).vec.back().first))){
            largest = pContext->threads->at(i).vec.back().first;
        }
    }
    return largest;
}


void waitForJob(JobHandle job){
    queue<IntermediateVec*>* queue =((JobContext*)(job))->shuffled;
    bool finished = ((JobContext*)(job))->shuffleFinished;
    WaitJobMutex mutex = ((JobContext*)(job))->jobMutex;
    if(!(queue->empty()  && finished)){ //job didnt finish
        if (pthread_mutex_lock(&mutex.mutex) != 0){
            fprintf(stderr, " error on pthread_mutex_lock");
            exit(1);
        }
        if (pthread_cond_wait(&mutex.cv, &mutex.mutex) != 0){ // then wait for broadcast
            fprintf(stderr, "error on pthread_cond_wait");
            exit(1);
        }
        if (pthread_mutex_unlock(&mutex.mutex) != 0){
            fprintf(stderr, "error on pthread_mutex_unlock");
            exit(1);
        }

    }

}

void getJobState(JobHandle job, JobState* state){
    JobContext* Job =((JobContext *) (job));

    //compute the current percentage
    if(Job->state->stage==MAP_STAGE){

        Job->state->percentage = (float)Job->precentege_atomic_counter/(float)Job->inputVec->size()*100;
    }
    else  if(Job->state->stage==REDUCE_STAGE){

        Job->state->percentage = (float)Job->atomic_counter/(float)Job->reduce_sum*100;
    }
    else {
        Job->state->percentage = 0;
    }
    state->stage = Job->state->stage;
    state->percentage = Job->state->percentage;
}

void closeJobHandle(JobHandle job) {
    JobContext* Job =((JobContext *) (job));
   if(false){
        fprintf(stderr, "Job didnt finish!");
        exit(1);
    }
    pthread_mutex_destroy(&(Job->shuff_mutex));
    pthread_mutex_destroy(&((JobContext *) (job))->queue_mutex);
    pthread_mutex_destroy(&((JobContext *) (job))->jobMutex.mutex);
    pthread_cond_destroy(&((JobContext *) (job))->jobMutex.cv);
    delete(((JobContext *) (job))->barrier);
    delete (((JobContext *) (job))->shuffled);
    delete(((JobContext*)(job))->state);
    delete(((JobContext*)(job))->threads);
    delete(((JobContext *) (job)));
}




