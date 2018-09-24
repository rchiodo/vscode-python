// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

import { commands, ExtensionContext, window } from 'vscode';

export class DataScience {
    public async activate(context: ExtensionContext): Promise<void> {
       this.registerCommands();
    }

    private registerCommands(): void {
        commands.registerCommand('python.datascience', this.executeDatascience);
    }

    private executeDatascience(): void {
        window.showInformationMessage('Hello Data Science');
    }
}
