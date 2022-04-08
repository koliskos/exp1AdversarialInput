# exp1AdversarialInput
First experiment of adversarial input project.

Essentially what we want to show is that when you smooth, you might be putting minorities (as defined above) have less of a right to indiviualism (and therefore can be at more of a disadvantage), since their data, which is outlier data, is disproportionately affected by the smoothing process. This means thatthey will be lumped into D (D = denial... defined below) instead of their deserved A outcome (A = approval... defined below). 

Can we measure the L2 norm between a bad output (ex: denial on a home loan application) and a better counterfactual (approval on a home loan) and show that for people in a minority (DEF {1} not minority as in marginalized group but minority as in an individual who may share features with a group that generally achieves outcome D, but they themselves deserve a different, less common output, A), the L2 norm is greater when the smoothing occurs vs the L2 norm is smaller (ie theoutput is closer to the better outcome for minority applicant) when smoothing doesnt occur, since the minorities are the ones smoothed over.

**Put more concisely, we will run experiemnt in the following way:**

1. Exp_1 will consist of training a model, M, on a dataset of homeloan applications. 

2. Run_min1 will consist of, taking a testing dataset called Test_unsmoothed, and identifying, or implanting into it, a distinct applicant datapoint, call it Min_A.  Min_A will be representative of a minority applicant (minority by DEF {1} ) who should be grouped within the desired outcome A. We will input Test_unsmoothed into M, and measure the distance, Dist_unsmoothed, between the outcome of Min_A and the nearest counterfactual (this counterfactual will be the nearest point for which Min_A's outcome is in A. Note: Dist_unsmoothed could be a negative number if Min_A's outcome is in A on Run_min1). 

We will then perform a smoothing technique on the Test_unsmoothed to make Test_smoothed. 

Then, for Run_min2, we will input Test_smoothed into M and measure the distance, Dist_smoothed, between the outcome of Min_A and the nearest counterfactual (this counterfactual will be the nearest point for which the outcome is opposite of Min_A). 

The hypothesis is that Dist_smoothed will be of greater magnitude than Dist_unsmoothed because point Min_A will have lost its unique credentials, which made it fit for an outcome of A, in the smoothing process. This would be because the smoothing process would directly target an individual like Min_A, who both has membership to a minority group and warrants an output distinct from fellow minority members. Its outcome point would be in (or further in) D after smoothing. This would effectively mean that Min_A was deprived of their right to be an individual (because of smoothing) (because of their membership to a minority group).

We will frame it more for minority (now minority by the definition of marginalized, not DEF {1} ) populations in the US, whose data may produce fewer positive outcomes given that the long standing, systematic oppression these populations have faced has made it less possible for individuals in these populations to embody data points which get classified into a desireable outcomes on decision problems. (Ex: GI bill makes it so white soldiers had the opportunity of home buying, while Black soldiers where not afforded this opportunity, which has present day repercussions on homeloan applications (ie, parents of a white applicant could provide a line of credit on their own owned home, while parents of a Black applicant who may be in the renting cycle are unable to provide that same type of line of credit.))
