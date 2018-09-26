package turtle

import (
	"bufio"
	"github.com/spaolacci/murmur3"
	"io"
	"math"
	"strconv"
	"strings"
)

const intercept = 11650396
const FNV_prime = 16777619

type ReadableModel struct {
	weights        []float32
	sparse         map[uint32]float32
	isSparse       bool
	bits           uint
	minLabel       float32
	maxLabel       float32
	oaa            uint32
	mask           uint32
	hashAll        bool
	multiClassBits uint32
	seed           uint32
	quadratic      map[byte][]byte
}

func parseInt(s string) int {
	n, _ := strconv.Atoi(s)
	return n
}

func parseFloat(s string) float32 {
	n, _ := strconv.ParseFloat(s, 32)
	return float32(n)
}

func NewReadableModel(r io.Reader) (*ReadableModel, error) {
	m := &ReadableModel{
		bits:      0,
		oaa:       1,
		mask:      0,
		sparse:    map[uint32]float32{},
		hashAll:   false,
		seed:      0,
		weights:   []float32{},
		quadratic: map[byte][]byte{},
	}
	scanner := bufio.NewScanner(r)
	inHeader := true
	nonZero := 0
	for scanner.Scan() {
		line := scanner.Text()

		splitted := strings.Split(line, ":")
		if inHeader {
			if strings.HasPrefix(line, "bits:") {
				m.bits = uint(parseInt(splitted[1]))
				m.weights = make([]float32, (1 << m.bits))
				m.mask = (1 << m.bits) - 1
			}
			if strings.HasPrefix(line, "Min label") {
				m.minLabel = parseFloat(splitted[1])
			}

			if strings.HasPrefix(line, "Max label") {
				m.maxLabel = parseFloat(splitted[1])
			}

			if strings.HasPrefix(line, "options:") {
				options := strings.Split(strings.Trim(splitted[1], " "), " ")
				for i := 0; i < len(options); i += 2 {
					value := options[i+1]
					if options[i] == "--oaa" {
						m.oaa = uint32(parseInt(options[i+1]))
						multiClassBits := uint32(0)
						for ml := uint32(m.oaa); ml > 0; ml = ml >> 1 {
							multiClassBits++
						}

						m.multiClassBits = multiClassBits
					} else if options[i] == "--quadratic" {
						m.quadratic[value[0]] = append(m.quadratic[value[0]], value[1])
					} else if options[i] == "--hash_seed" {
						m.seed = uint32(parseInt(value))
					} else if options[i] == "--hash" {
						if value == "all" {
							m.hashAll = true
						}
					}
				}
			}
			if line == ":0" {
				inHeader = false
			}
		} else {
			m.weights[parseInt(splitted[0])] = parseFloat(splitted[1])
			nonZero++
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	// random threshold that made sense when i was thinking about it
	// XXX; do some benchmarks when it makes sense, or make it configurable
	if nonZero < int(m.mask/8) {
		m.isSparse = true
		for i, w := range m.weights {
			m.sparse[uint32(i)] = w
		}
		m.weights = nil
	}
	return m, nil
}

type Feature struct {
	Value          float32
	Name           string
	NameInt        uint32
	hash           uint32
	hashIsComputed bool
}

type Namespace struct {
	Name           string
	hash           uint32
	hashIsComputed bool
	Features       []*Feature
}

type Request struct {
	Namespaces    []*Namespace
	Probabilities bool
}

func NewFeatureString(name string, value float32) *Feature {
	return &Feature{
		Name:  name,
		Value: value,
	}
}

func NewFeatureInt(name uint32, value float32) *Feature {
	return &Feature{
		NameInt: name,
		Value:   value,
	}
}

func NewNamespace(name string, fs ...*Feature) *Namespace {
	return &Namespace{Name: name, Features: fs}
}
func NewRequest(ns ...*Namespace) *Request {
	return &Request{
		Namespaces: ns,
	}
}
func (m *ReadableModel) getBucket(featureHash uint32, klass uint32) uint32 {
	return ((featureHash << m.multiClassBits) | klass) & m.mask
}

func Identity(a float32) float32 {
	return a
}
func Logistic(a float32) float32 {
	return float32((1. / (1. + math.Exp(float64(-a)))))
}
func (m *ReadableModel) get(bucket uint32) float32 {
	if m.isSparse {
		v, ok := m.sparse[bucket]
		if ok {
			return v
		}
		return 0
	} else {
		return m.weights[bucket]
	}
}
func (m *ReadableModel) Predict(req *Request) []float32 {
	out := make([]float32, m.oaa)
	for _, ns := range req.Namespaces {
		if !ns.hashIsComputed {
			if len(ns.Name) == 0 {
				ns.hash = 0
			} else {
				ns.hash = murmur3.Sum32WithSeed([]byte(ns.Name), m.seed)
			}
		}

		for _, f := range ns.Features {
			if !f.hashIsComputed {
				if len(f.Name) == 0 {
					if m.hashAll {
						f.hash = murmur3.Sum32WithSeed([]byte(strconv.FormatUint(uint64(f.NameInt), 10)), ns.hash)
					} else {

						f.hash = f.NameInt + ns.hash
					}

				} else {
					f.hash = murmur3.Sum32WithSeed([]byte(f.Name), ns.hash)
				}
			}

			for klass := uint32(0); klass < m.oaa; klass++ {
				bucket := m.getBucket(f.hash, klass)
				out[klass] += f.Value * m.get(bucket)
			}

		}
	}

	if len(m.quadratic) > 0 {
		for startingWith, interactingWith := range m.quadratic {

		NEXT:
			for _, nsA := range req.Namespaces {
				if nsA.Name[0] != startingWith {
					continue NEXT
				}
				for _, inter := range interactingWith {
					for _, nsB := range req.Namespaces {
						if nsB.Name[0] == inter {
							for _, featureA := range nsA.Features {
								for _, featureB := range nsB.Features {
									fnv := ((featureA.hash * FNV_prime) ^ featureB.hash)
									for klass := uint32(0); klass < m.oaa; klass++ {
										bucket := m.getBucket(fnv, klass)
										out[klass] += featureA.Value * featureB.Value * m.get(bucket)
									}
								}
							}
						}
					}
				}
			}
		}
	}

	for klass := uint32(0); klass < m.oaa; klass++ {
		bucket := m.getBucket(intercept, klass)
		out[klass] += m.get(bucket)
		out[klass] = clipTo(out[klass], m.minLabel, m.maxLabel)
	}
	if req.Probabilities {
		for klass := uint32(0); klass < m.oaa; klass++ {
			out[klass] = Logistic(out[klass])
		}

		if m.oaa > 1 {
			sum := float32(0)
			for klass := uint32(0); klass < m.oaa; klass++ {
				sum += out[klass]
			}
			for klass := uint32(0); klass < m.oaa; klass++ {
				out[klass] = out[klass] / sum
			}
		}
	}

	return out
}
func max(x, y float32) float32 {
	if x > y {
		return x
	}
	return y
}
func min(x, y float32) float32 {
	if x < y {
		return x
	}
	return y
}
func clipTo(x, mmin, mmax float32) float32 {
	return max(min(x, mmax), mmin)
}
