// Copyright 2022 The Xorm Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utils

import (
	"fmt"

	"xorm.io/builder"
)

type BuildReader interface {
	String() string
	Args() []interface{}
}

// WriteBuilder writes writers to one
func WriteBuilder(w *builder.BytesWriter, inputs ...BuildReader) error {
	for _, input := range inputs {
		if _, err := fmt.Fprint(w, input.String()); err != nil {
			return err
		}
		w.Append(input.Args()...)
	}
	return nil
}
