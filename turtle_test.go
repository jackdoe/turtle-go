package turtle

import (
	"strings"
	"testing"
)

const quadratic = `
Version 8.6.1
Id 
Min label:-1
Max label:1
bits:18
lda:0
0 ngram:
0 skip:
options: --hash_seed 0 --quadratic ab --link identity
Checksum: 3417833913
:0
6163:0.0624969
7472:-0.12023
42847:-0.0421919
49960:-0.12023
51692:0.0624969
55432:-0.0421919
85288:-0.0421919
97903:-0.0421919
105687:-0.0421919
106615:-0.0421919
116060:-0.0421919
133893:-0.0421919
145317:-0.0421919
187130:-0.0421919
190909:-0.0421919
202246:-0.0421919
221854:0.0624969
223322:-0.12023
244133:-0.12023
244870:0.0624969
`

const oaa = `
Version 8.6.1
Id 
Min label:-1
Max label:1
bits:18
lda:0
0 ngram:
0 skip:
options: --hash_seed 0 --oaa 3 --link identity
Checksum: 4199872739
:0
47580:-0.274889
47581:-0.27124
47582:0.285907
138032:0.253423
138033:-0.253423
138034:-0.253423
154112:-0.277909
154113:0.277909
154114:-0.247965
202096:-0.113007
202097:-0.175926
202098:-0.188205
`

func TestQuadratic(t *testing.T) {
	m, err := NewReadableModel(strings.NewReader(quadratic))

	if err != nil {
		t.Error(err)
	}
	if len(m.quadratic) != 1 {
		t.Errorf("expected %d got %d", 1, len(m.quadratic))
	}
	req := NewRequest(NewNamespace("a", NewFeatureString("x", 1), NewFeatureString("z", 1)), NewNamespace("b", NewFeatureString("x1", 1.0), NewFeatureString("z1", 1.0)))
	pred := m.Predict(req)

	if pred[0] != -0.0656607 {
		t.Errorf("unable to predict %#v", pred)
	}
}

func TestOaa(t *testing.T) {
	m, _ := NewReadableModel(strings.NewReader(oaa))
	if m.oaa != 3 {
		t.Errorf("expected %d got %d", 3, m.oaa)
	}

	//[0.140416, -0.429349, -0.441628],
	req := NewRequest(NewNamespace("",
		NewFeatureString("pos", 1),
		NewFeatureString("pos", 1),
		NewFeatureString("pos", 1),
		NewFeatureString("pos", 1),
		NewFeatureString("pos", 1),
		NewFeatureString("pos", 1),
		NewFeatureString("pos", 1)))
	pred := m.Predict(req)

	if pred[0] != -0.113007 || pred[1] != -0.175926 || pred[2] != -0.188205 {
		t.Errorf("unable to predict %#v", pred)
	}

	req.Probabilities = true
	pred = m.Predict(req)
	if pred[0] != 0.34162152 || pred[1] != 0.3302915 || pred[2] != 0.328087 {
		t.Errorf("unable to predict %#v", pred)
	}

}
