#include <iostream>   // Για είσοδο/έξοδο
#include <vector>     // Για χρήση δομής δεδομένων vector
#include <cmath>      // Για μαθηματικές συναρτήσεις όπως sqrt και pow
#include <cstdlib>    // Για τυχαίους αριθμούς
#include <ctime>      // Για seed των τυχαίων αριθμών
#include <fstream>    // Για εγγραφή αποτελεσμάτων σε αρχείο
#include <functional> // Για χρήση τύπου std::function

using namespace std; // Χρησιμοποιούμε το namespace std για απλοποίηση

// Ορισμός παραμέτρων του αλγορίθμου
const int MAX_ITERATIONS = 4;         // Μέγιστος αριθμός επαναλήψεων
const double LEARNING_RATE = 0.001;  // Ρυθμός μάθησης για Gradient Descent
const double EPSILON = 1e-6;        // Όριο ακρίβειας (συγκλίνουσα συνάρτηση)
const int SAMPLE_SIZE = 5;         // Αριθμός τυχαίων σημείων ανά επανάληψη
const int DIMENSIONS = 2;         // Διάσταση των συναρτήσεων (π.χ., 2 για x1, x2)

// Τύποι για τις συναρτήσεις και τα gradients τους
using Function = function<double(const vector<double>&)>;  // Ορισμός τύπου για τις συναρτήσεις αντικειμενικού σκοπού
using Gradient = function<vector<double>(const vector<double>&)>;  // Ορισμός τύπου για τα gradients των συναρτήσεων

// Συνάρτηση Camel (γνωστή συνάρτηση με πολλαπλά τοπικά ελάχιστα)
double camelFunction(const vector<double>& x) {
    double x1 = x[0], x2 = x[1];
    return 4*x1*x1 - 2.1*pow(x1, 4) + pow(x1, 6)/3.0 + x1*x2 - 4*x2*x2 + 4*pow(x2, 4);
}

// Gradient της συνάρτησης Camel
vector<double> camelGradient(const vector<double>& x) {
    vector<double> grad(DIMENSIONS);  // Δημιουργία ενός διανύσματος για το gradient
    double x1 = x[0], x2 = x[1];
    grad[0] = 8*x1 - 8.4*pow(x1, 3) + 2*pow(x1, 5) + x2; // Υπολογισμός του gradient ως προς x1
    grad[1] = x1 - 8*x2 + 16*pow(x2, 3); // Υπολογισμός του gradient ως προς x2
    return grad; // Επιστροφή του gradient
}

// Συνάρτηση Rastrigin (γνωστή συνάρτηση με πολλά τοπικά ελάχιστα)
double rastriginFunction(const vector<double>& x) {
    double result = 10 * DIMENSIONS; // Η τιμή της συνάρτησης με πολλαπλά τοπικά ελάχιστα
    for (double xi : x) {
        result += xi * xi - 10 * cos(2 * M_PI * xi);  // Προσθήκη όρων από τη συνάρτηση Rastrigin
    }
    return result; // Επιστροφή της τιμής της συνάρτησης
}

// Gradient της συνάρτησης Rastrigin
vector<double> rastriginGradient(const vector<double>& x) {
    vector<double> grad(DIMENSIONS);  // Δημιουργία διανύσματος για το gradient
    for (size_t i = 0; i < x.size(); i++) {
        grad[i] = 2 * x[i] + 20 * M_PI * sin(2 * M_PI * x[i]); // Υπολογισμός του gradient
    }
    return grad; // Επιστροφή του gradient
}

// Συνάρτηση Griewank (γνωστή συνάρτηση βελτιστοποίησης με πολλαπλά τοπικά ελάχιστα)
double griewankFunction(const vector<double>& x) {
    double sum = 0.0, prod = 1.0; // Αρχικοποίηση αθροίσματος και γινομένου
    for (size_t i = 0; i < x.size(); i++) {
        sum += pow(x[i], 2) / 4000.0; // Υπολογισμός αθροίσματος
        prod *= cos(x[i] / sqrt(i + 1)); // Υπολογισμός γινομένου
    }
    return 1 + sum - prod; // Επιστροφή της τιμής της συνάρτησης Griewank
}

// Gradient της συνάρτησης Griewank
vector<double> griewankGradient(const vector<double>& x) {
    vector<double> grad(DIMENSIONS); // Δημιουργία διανύσματος για το gradient
    for (size_t i = 0; i < x.size(); i++) {
        grad[i] = x[i] / 2000.0 - sin(x[i] / sqrt(i + 1)) / sqrt(i + 1); // Υπολογισμός gradient
    }
    return grad; // Επιστροφή του gradient
}

// Συνάρτηση Branin (γνωστή συνάρτηση βελτιστοποίησης με γνωστά τοπικά ελάχιστα)
double braninFunction(const vector<double>& x) {
    double x1 = x[0], x2 = x[1];
    return pow(x2 - 5.1 / (4 * M_PI * M_PI) * pow(x1, 2) + 5 / M_PI * x1 - 6, 2)
           + 10 * (1 - 1 / (8 * M_PI)) * cos(x1) + 10;  // Υπολογισμός της συνάρτησης Branin
}

// Gradient της συνάρτησης Branin
vector<double> braninGradient(const vector<double>& x) {
    vector<double> grad(DIMENSIONS); // Δημιουργία διανύσματος για το gradient
    double x1 = x[0], x2 = x[1];
    grad[0] = - 2 * x1 * (5.1 / (4 * M_PI * M_PI)) + 5 / M_PI * sin(x1) + 2 * (x2 - 5.1 / (4 * M_PI * M_PI) * pow(x1, 2));
    grad[1] = 2 * (x2 - 5.1 / (4 * M_PI * M_PI) * pow(x1, 2)) + 10 * (1 - 1 / (8 * M_PI)) * sin(x1); // Υπολογισμός του gradient
    return grad; // Επιστροφή του gradient
}

// Συνάρτηση Shubert (γνωστή συνάρτηση με πολλαπλά τοπικά ελάχιστα)
double shubertFunction(const vector<double>& x) {
    double result = 0.0;
    for (int i = 1; i <= 5; i++) {
        result += i * sin(i * x[0] + i) * sin(i * x[1] + i); // Υπολογισμός της συνάρτησης Shubert
    }
    return -result; // Επιστροφή αρνητικής τιμής για τοπικά ελάχιστα
}

// Gradient της συνάρτησης Shubert
vector<double> shubertGradient(const vector<double>& x) {
    vector<double> grad(DIMENSIONS); // Δημιουργία διανύσματος για το gradient
    grad[0] = 0.0;
    grad[1] = 0.0;
    for (int i = 1; i <= 5; i++) {
        grad[0] += i * cos(i * x[0] + i) * sin(i * x[1] + i); // Υπολογισμός του gradient ως προς x1
        grad[1] += i * sin(i * x[0] + i) * cos(i * x[1] + i); // Υπολογισμός του gradient ως προς x2
    }
    return grad; // Επιστροφή του gradient
}

// Συνάρτηση για τον έλεγχο NaN
bool isNaN(double value) {
    return value != value; // Αν η τιμή είναι NaN, επιστρέφει true
}

// Gradient Descent για τοπική ελαχιστοποίηση
vector<double> gradientDescent(const vector<double>& startPoint, const Function& func, const Gradient& gradFunc) {
    vector<double> x = startPoint;

    // Επανάληψη μέχρι τη σύγκλιση ή το μέγιστο αριθμό επαναλήψεων
    for (int iter = 0; iter < 1000; iter++) {
        vector<double> grad = gradFunc(x); // Υπολογισμός του gradient
        double norm = 0.0; // Υπολογισμός του μέτρου του gradient
        for (double g : grad) norm += g * g;

        if (sqrt(norm) < EPSILON) break; // Αν το μέτρο είναι πολύ μικρό, σταματάμε

        for (size_t i = 0; i < x.size(); i++) {
            x[i] -= LEARNING_RATE * grad[i]; // Ενημέρωση της θέσης του ελάχιστου
        }
    }

    return x; // Επιστροφή του τοπικού ελαχίστου
}

// Έλεγχος αν δύο σημεία είναι κοντά
bool arePointsClose(const vector<double>& a, const vector<double>& b, double threshold = EPSILON) {
    double distance = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        distance += pow(a[i] - b[i], 2); // Υπολογισμός της Ευκλείδειας απόστασης
    }
    return sqrt(distance) < threshold; // Αν η απόσταση είναι μικρότερη από το threshold, επιστρέφει true
}

// MinFinder αλγόριθμος
void minFinder(const Function& func, const Gradient& gradFunc, ofstream& output, const string& functionName) {
    vector<vector<double>> localMinima; // Λίστα με τα τοπικά ελάχιστα
    srand(static_cast<unsigned>(time(0))); // Αρχικοποίηση του seed για τυχαίους αριθμούς

    output << "-------------------------------------------" << endl;
    output << "Εκτέλεση για τη συνάρτηση: " << functionName << endl;
    output << "-------------------------------------------" << endl;

    // Εκτέλεση της στοχαστικής διαδικασίας για την εύρεση τοπικών ελαχίστων
    for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
        for (int i = 0; i < SAMPLE_SIZE; i++) {
            vector<double> point(DIMENSIONS);  // Δημιουργία τυχαίου σημείου

            // Δημιουργία τυχαίων τιμών για κάθε συντεταγμένη
            for (int d = 0; d < DIMENSIONS; d++) {
                point[d] = -5.0 + static_cast<double>(rand()) / RAND_MAX * 10.0;
            }

            // Εύρεση τοπικού ελαχίστου μέσω Gradient Descent
            vector<double> minimum = gradientDescent(point, func, gradFunc);

            // Έλεγχος αν η τιμή της συνάρτησης είναι NaN
            if (isNaN(func(minimum))) {
                continue; // Αν το αποτέλεσμα είναι NaN, το παραλείπουμε
            }

            // Έλεγχος αν είναι νέο τοπικό ελάχιστο
            bool isNewMinimum = true;
            for (const auto& found : localMinima) {
                if (arePointsClose(minimum, found)) {
                    isNewMinimum = false; // Αν υπάρχει ήδη το σημείο, το παραλείπουμε
                    break;
                }
            }

            // Αν είναι νέο τοπικό ελάχιστο, το καταγράφουμε
            if (isNewMinimum) {
                localMinima.push_back(minimum);
                output << "Βρέθηκε νέο τοπικό ελάχιστο: ";
                for (double xi : minimum) output << xi << " "; // Εμφάνιση των συντεταγμένων
                output << "f(x) = " << func(minimum) << endl; // Εμφάνιση της τιμής της συνάρτησης
            }
        }
    }
}

// Κύριο πρόγραμμα
int main() {
    ofstream output("results_minfinder.txt"); // Δημιουργία του αρχείου εξόδου
    if (!output.is_open()) {
        cerr << "Αποτυχία ανοίγματος αρχείου εξόδου." << endl; // Αν αποτύχει το άνοιγμα του αρχείου
        return 1;
    }

    // Εκτέλεση του αλγορίθμου για τις διάφορες συναρτήσεις
    minFinder(camelFunction, camelGradient, output, "Camel");
    minFinder(rastriginFunction, rastriginGradient, output, "Rastrigin");
    minFinder(griewankFunction, griewankGradient, output, "Griewank");
    minFinder(braninFunction, braninGradient, output, "Branin");
    minFinder(shubertFunction, shubertGradient, output, "Shubert");

    output.close(); // Κλείσιμο του αρχείου
    cout << "Τα αποτελέσματα γράφτηκαν στο αρχείο: results_minfinder.txt" << endl; // Εκτύπωση μηνύματος
    return 0;
}
