/////////////////////////////////////////////////////////////////////////////////////////////
// LIBRARY - stochastic gradient descent solver for SVM
// by Michal Hradis 2011
// Based on work of Leon Bottou http://leon.bottou.org/projects/sgd
// Licence: Lesser General Public License http://www.gnu.org/licenses/lgpl.html
/////////////////////////////////////////////////////////////////////////////////////////////


#include "sgdSVM.h"

#include <cstring> 
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <algorithm>

template< class _T>
_T dot( const _T *vec1, const _T *vec2,  const int length)
{
	_T sum = (_T)0;
	for( int i = 0; i < length; i++){
		sum += vec1[ i] * vec2[ i];
	}
	return sum;
}

template< class _T>
void scale( _T *vec1, const int length, _T scale)
{
	for( int i = 0; i < length; i++){
		vec1[ i] *= scale;
	}
}

template< class _T>
void add( _T *vec1, const _T *vec2, const int length, _T scale)
{
	for( int i = 0; i < length; i++){
		vec1[ i] += vec2[ i] * scale;
	}
}



using namespace std;



inline double loss( const double z, const TLoss lossType)
{
	switch ( lossType){
		case LOGLOSS:
			if (z > 18)
				return exp(-z);
			if (z < -18)
				return -z;
			return log(1+exp(-z));

		case LOGLOSSMARGIN:
			if (z > 18)
				return exp(1-z);
			if (z < -18)
				return 1-z;
			return log(1+exp(1-z));

		case SMOOTHHINGELOSS:
			if (z < 0)
				return 0.5 - z;
			if (z < 1)
				return 0.5 * (1-z) * (1-z);
			return 0;

		case SQUAREDHINGELOSS:
			if (z < 1)
				return 0.5 * (1 - z) * (1 - z);
			return 0;

		case HINGELOSS:
			if (z < 1)
				return 1 - z;
			return 0;

		case EXPLOSS:
			return exp( -z);

		default:
			cerr << "Undefined loss " << lossType << endl;
			exit(-1);
	}
}

inline float loss( const float z, const TLoss lossType)
{
	switch ( lossType){
		case LOGLOSS:
			if (z > 18.0f)
				return expf(-z);
			if (z < -18.0f)
				return -z;
			return logf( 1.0f + expf( -z));

		case LOGLOSSMARGIN:
			if (z > 18.0f)
				return expf( 1.0f - z);
			if (z < -18.0f)
				return 1.0f - z;
			return logf( 1.0f + expf( 1.0f - z));

		case SMOOTHHINGELOSS:
			if (z < 0.0f)
				return 0.5f - z;
			if (z < 1.0f)
				return 0.5f * (1.0f - z) * ( 1.0f - z);
			return 0.0f;

		case SQUAREDHINGELOSS:
			if (z < 1.0f)
				return 0.5f * (1.0f - z) * (1.0f - z);
			return 0.0f;

		case HINGELOSS:
			if (z < 1.0f)
				return 1.0f - z;
			return 0.0f;

		case EXPLOSS:
			return expf( -z);

		default:
			cerr << "Undefined loss " << (int)lossType << endl;
			exit(-1);
	}
}

inline double dloss(double z, const TLoss lossType)
{
	switch ( lossType){
		case LOGLOSS:
			if (z > 18)
				return exp(-z);
			if (z < -18)
				return 1;
			return 1 / (exp(z) + 1);
		case LOGLOSSMARGIN:
			if (z > 18)
				return exp(1-z);
			if (z < -18)
				return 1;
			return 1 / (exp(z-1) + 1);
		case SMOOTHHINGELOSS:
			if (z < 0)
				return 1;
			if (z < 1)
				return 1-z;
			return 0;
		case SQUAREDHINGELOSS:
			if (z < 1)
				return (1 - z);
			return 0;

		case EXPLOSS:
			return exp( -z);

		default:
			if (z < 1)
				return 1;
			return 0;
	}
}

inline float dloss( float z, const TLoss lossType)
{
	switch ( lossType){
		case LOGLOSS:
			if (z > 18.0f)
				return expf( -z);
			if (z < -18.0f)
				return 1;
			return 1.0f / ( expf(z) + 1.0f);
		case LOGLOSSMARGIN:
			if (z > 18.0f)
				return expf( 1 - z);
			if (z < -18.0f)
				return 1.0f;
			return 1.0f / ( expf( z - 1.0f) + 1.0f);
		case SMOOTHHINGELOSS:
			if (z < 0.0f)
				return 1.0f;
			if (z < 1.0f)
				return 1 - z;
			return 0.0f;
		case SQUAREDHINGELOSS:
			if (z < 1.0f)
				return (1.0f - z);
			return 0.0f;

		case EXPLOSS:
			return expf( -z);

		default:
			if (z < 1.0f)
				return 1.0f;
			return 0.0f;
	}
}


void prepareLossTranslation()
{
	lossTranslation[ ""] = HINGELOSS;
	lossTranslation[ "HINGELOSS"] = HINGELOSS;
	lossTranslation[ "SMOOTHHINGELOSS"] = SMOOTHHINGELOSS;
	lossTranslation[ "SQUAREDHINGELOSS"] = SQUAREDHINGELOSS;
	lossTranslation[ "LOGLOSS"] = LOGLOSS;
	lossTranslation[ "LOGLOSSMARGIN"] = LOGLOSSMARGIN;
	lossTranslation[ "EXPLOSS"] = EXPLOSS;
	for( std::map< string, TLoss>::const_iterator it = lossTranslation.begin(); it != lossTranslation.end(); it++){
		invLossTranslation[ it->second] = it->first;
	}
}

// -- stochastic gradient

vector< float> TSVMSgd::getW(){
	vector< float> W( dimension + 1);
	for( int j = 0; j < dimension; j++){
		W[ j] = w[ j];
	}
	W.back() = bias;

	return W;
}


void TSVMSgd::setClassifier( const TFloatRep *_w, const double _bias)
{
	memcpy( w, _w, sizeof( TFloatRep) * dimension);
	bias = _bias;
}

TSVMSgd::TSVMSgd( const int dim, const TFloatRep _lambda, const std::string &_loss)
	: lambda( _lambda), w( new TFloatRep[ dim]), wscale(1), bias(0), dimension( dim)
{
	prepareLossTranslation();

	loss = lossTranslation[ _loss];

	for( int k = 0; k < dimension; k++){
		w[k] = 0;
	}

	// Shift t in order to have a 
	// reasonable initial learning rate.
	// This assumes |x| \approx 1.
	const double maxw = 1.0 / sqrt(lambda);
	const double typw = sqrt(maxw);
	const double eta0 = typw / max( 1.0, dloss( -typw, loss));
	t = (TFloatRep)( 1 / (eta0 * lambda));
}

void TSVMSgd::train( const std::vector< TFloatRep *> &data, const vector< TFloatRep> &labels)
{
	vector< TFloatRep *> localData( data);
	vector< TFloatRep> localLabels( labels);

	while( !localData.empty()){

		const int ID = rand() % localData.size();

		TFloatRep* x = localData[ ID];
		double y = localLabels[ ID];

		localData[ ID] = localData.back(); localData.pop_back();
		localLabels[ ID] = localLabels.back(); localLabels.pop_back();


		const double eta = 1.0 / (lambda * t);
		const double s = 1 - eta * lambda;
		wscale *= s;
		if (wscale < 1e-9){
			scale( w, dimension,(TFloatRep)( wscale));
			wscale = 1;
		}

		const double wx = dot( w, x, dimension) * wscale;

		const double z = y * (wx + bias);

		if( loss >= LOGLOSS || z < 1){

			const double etd = eta * dloss( z, loss);
			add( w, x, dimension, (TFloatRep)( etd * y / wscale));

			bias += etd * y * 0.01; // Slower rate on the bias because it learns at each iteration.
		}
		t += 1;
	}

	scale( w, dimension,(TFloatRep)( wscale));
	wscale = 1;

	const double wnorm =  dot( w, w, dimension) * wscale * wscale;
	cout << " Norm: " << wnorm << ", Bias: " << bias << endl;
}

double TSVMSgd::eval( TFloatRep * x)
{
	return dot( x, w, dimension) + bias;

}


void TSVMSgd::eval( const std::vector< TFloatRep *> &x, std::vector< TFloatRep> &y)
{
	for( int i = 0; i < (int)x.size(); i++)
	{
		y[i] = dot( x[ i], w, dimension) + bias;
	}
}



double dot( double *vec1, const unsigned char *vec2,  const int dim, const int step)
{
	double sum = 0;
	for( int i = 0; i < dim; i++, vec1 += step){
		sum += vec1[ vec2[i]];
	}
	return sum;
}

void add( double *vec1, const unsigned char *vec2, const int dim, const int step, double scale)
{
	for( int i = 0; i < dim; i++, vec1 += step){
		vec1[ vec2[i]] += scale;
	}
}



TSparseCodeSGD::TSparseCodeSGD( const int _dimension, const int _step, const double _lambda, const std::string &_loss)
	: dimension( _dimension), step( _step), w( dimension * step, 0), lambda( _lambda),   wscale( 1), bias( 0)
{
	assert( dimension > 0);
	assert( step > 1);
	assert( step <= 256);
	assert( lambda >= 0);

	prepareLossTranslation();

	loss = lossTranslation[ _loss];

	// Shift t in order to have a 
	// reasonable initial learning rate.
	// This assumes |x| \approx 1.
	const double maxw = 1.0 / sqrt(lambda);
	const double typw = sqrt(maxw);
	const double eta0 = typw / max( 1.0, dloss( -typw, loss));
	if( loss != EXPLOSS){
		t = 1 / (eta0 * lambda);
		t = max( 2.0, t);
	} else {
		t = 1 / lambda;
	}
}


vector< double> TSparseCodeSGD::getW()
{ 		
	scale( &w[0], dimension, wscale);
	wscale = 1;
	return w;
}


void TSparseCodeSGD::update( const unsigned char *x, const double classID)
{
	if( lambda == 0){
		const double eta = 1.0 / (t);

		const double wx = dot( &w[0], x, dimension, step) / dimension;

		const double z = ((classID > 0)?(+1.0):(-1.0)) * (wx + bias);

		const double etd = eta * dloss( z, loss);
		if( etd > 5){
			cerr << "ETD: " << etd << endl;
		}
		add( &w[0], x, dimension, step, etd * classID / dimension);

		bias += etd * classID * 0.01; // Slower rate on the bias because it learns at each iteration.

	} else {
		const double eta = 1.0 / (lambda * t);
		const double s = 1 - abs( classID) * eta * lambda;
		wscale *= s;
		if (wscale < 1e-9){
			scale( &w[0], dimension, wscale);
			wscale = 1;
		}

		const double wx = dot( &w[0], x, dimension, step) / dimension * wscale;

		const double z = ((classID > 0)?(+1.0):(-1.0)) * (wx + bias);

		if( loss >= LOGLOSS || z < 1){

			const double etd = min( 10.0, eta * dloss( z, loss));

			if( etd > 5){
				cerr << "ETD: " << etd << endl;
			}
			add( &w[0], x, dimension, step, etd * classID / wscale / dimension);

			bias += etd * classID * 0.01; // Slower rate on the bias because it learns at each iteration.
		}
	}
	t += 1;
}

double TSparseCodeSGD::eval( const unsigned char *x)
{
	return dot( &w[0], x, dimension, step) / dimension + bias;
}

TTestHistogram::TTestHistogram( const vector< double> &w, const double bias, int dim)
	: minVal( -1e-5), maxVal( 1e-5), posHistogram( binCount), negHistogram( binCount)
{
	assert( w.size() % dim == 0);

	int step = w.size() / dim;

	for( int i = 0; i < dim; i++){
		double maxV = -1e50;
		double minV = +1e50;

		for( int j = 0; j < step; j++){
			maxV = max( maxV, w[ i * step + j]);
			minV = min( minV, w[ i * step + j]);
		}

		minVal += minV;
		maxVal += maxV;
	}

	minVal += bias;
	maxVal += bias;

	minVal *= 0.01;
	maxVal *= 0.01;
}

void TTestHistogram::accumulate( const double val, bool pos)
{
	int index = (int)((val - minVal) / (maxVal - minVal) *( binCount - 1) + 0.5);

	index = max( 0, min( binCount - 1, index));

	if( pos){
		posHistogram[ index] += 1;
	} else {
		negHistogram[ index] += 1;
	}
}

void TTestHistogram::compute( double &avgP, double &eer, double &negRejection, double & posRejection, double & threshold, const double alpha)
{

	double posCount = 0;
	double negCount = 0;
	
	avgP = 0;

	for( int i = binCount - 1; i >= 0; i--){

		negCount += negHistogram[ i];
		posCount += posHistogram[ i];

		if( posCount > 0){
			avgP += posCount / ( posCount + negCount) * posHistogram[ i];
		}
	}
	avgP /= posCount;


	double fp = negCount;
	double fn = 0;
	eer = 1.0;

	for( int i = 0; i < binCount; i++){
		fn += posHistogram[ i];
		fp -= negHistogram[ i];

		if( fn / posCount >= fp / negCount){
			eer =  (fn / posCount + fp / negCount) / 2;
			break;
		}
	}

	negRejection = 0;
	posRejection = 0;
	double cumPos = 0;
	double cumNeg = 0;
	double A = 1.0 / alpha;

	for( int i = 0; i < binCount; i++){
		cumPos += posHistogram[ i];
		cumNeg += negHistogram[ i];
		if( (cumNeg / negCount) / (cumPos / posCount) >= A){
			negRejection = cumNeg / negCount;
			posRejection = cumPos / posCount;
			threshold = i / (double) binCount * ( maxVal - minVal) + minVal;
		}
	}
}

void TTestHistogram::getHistograms( std::vector< double > & positive, std::vector< double > & negative, std::vector< double > & xAxis)
{
	positive = posHistogram;
	negative = negHistogram;
	xAxis.resize( binCount);

	for( int i = 0; i < binCount; i++){
		xAxis[ i] = i / (double) binCount * ( maxVal - minVal) + minVal;
	}
}
