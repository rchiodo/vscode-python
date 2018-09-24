// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

import { ExtensionContext } from 'vscode';

export const IDataScience = Symbol('IDataScience');
export interface IDataScience {
    activate(context: ExtensionContext): Promise<void>;
}
