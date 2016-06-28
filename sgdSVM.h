/////////////////////////////////////////////////////////////////////////////////////////////
// LIBRARY - stochastic gradient descent solver for SVM
// by Michal Hradis 2011
// Based on work of Leon Bottou http://leon.bottou.org/projects/sgd
// Licence: Lesser General Public License http://www.gnu.org/licenses/lgpl.html
/////////////////////////////////////////////////////////////////////////////////////////////


#ifndef __sgdSVM_h__
#define __sgdSVM_h__

#include <vector>
#include <string>
#include <map>

typedef double TFloatRep;

enum TLoss { HINGELOSS = 1, SMOOTHHINGELOSS = 2, SQUAREDHINGELOSS = 3, LOGLOSS = 10, LOGLOSSMARGIN = 11, EXPLOSS = 12};

static std::map< std::string, TLoss> lossTranslation;
static std::map< TLoss, std::string> invLossTranslation;

void prepareLossTranslation();

class TSVMSgd
{

	const int dimension;
	TFloatRep *w;

	double  t;
	double lambda;
	double wscale;
	double bias;
	TLoss loss;



public:

	TSVMSgd( const int dim, const TFloatRep lambda, const std::string &_loss = "HINGELOSS");

	void setClassifier( const TFloatRep *_w, const double _bias);

	void train( const std::vector< TFloatRep *> &x, const std::vector< TFloatRep> &y);

	void eval( const std::vector< TFloatRep *> &x, std::vector< TFloatRep> &y);

	double eval( TFloatRep * x);

	std::vector< float> getW();
};



class TSparseCodeSGD
{

	int dimension;
	int step;
	std::vector< double> w;

	double  t;
	double lambda;
	double wscale;
	double bias;
	TLoss loss;

public:

	TSparseCodeSGD( const int dim, const int step, const double lambda, const std::string &_loss = "HINGELOSS");

	void update( const unsigned char *x, const double classID);

	double eval( const unsigned char *x);

	std::vector< double> getW();

	double getBias(){ return bias;}


};



class TTestHistogram
{

	static const int binCount = 2000;

	double minVal;
	double maxVal;
	std::vector< double> posHistogram;
	std::vector< double> negHistogram;

public:

	TTestHistogram( const std::vector< double> &w, const double bias, int dim);

	void accumulate( const double val, bool pos);

	void compute( double &avgP, double &eer, double &negRejection, double & posRejection, double & threshold, const double alpha = 0.002);

	void getHistograms( std::vector< double > & positive, std::vector< double > & negative, std::vector< double > & xAxis);
};


#endif