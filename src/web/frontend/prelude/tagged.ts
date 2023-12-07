export interface Tagged<a extends string> {
  _tag: a;
}

export type Untagged<a extends Tagged<string>> = Omit<a, "_tag">;
